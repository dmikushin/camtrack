// XXX Still doesn't work in Discord

// sudo modprobe v4l2loopback exclusive_caps=1
// NOT USED: v4l2loopback-ctl set-caps video/x-raw,format=UYVY,width=640,height=480 /dev/video0

#include <cassert>
#include <cmath>
#include <iostream>

#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <err.h>
#include <fcntl.h>
#include <unistd.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>

using std::cout;
using std::cerr;
using std::endl;

typedef float prec;
constexpr int kOutWidth = 640;
constexpr int kOutHeight = 360;
constexpr prec kOutAspect = (prec)kOutWidth / kOutHeight;
// I put a lot of primes in here to try to keep the zoom algorithms
// from converging on small binary fractions; I figured that would
// allow the image smoothing algorithms to blur out shifts.  Sadly, it
// didn't work.
constexpr prec kLPFSlewRate = 1.0/101;
constexpr prec kBoundedMaxSlew = 43.0/41.0;
constexpr prec kBoundedMaxSlewAccel = 1.0/13;

inline void
dbgRect(cv::Mat &img __attribute__((unused)),
        cv::Rect rec __attribute__((unused)),
        const cv::Scalar &color __attribute__((unused)),
        int thickness __attribute__((unused)) = 1,
        int lineType __attribute__((unused)) = cv::LINE_8,
        int shift __attribute__((unused)) = 0) {
  cv::rectangle(img, rec, color, thickness, lineType, shift);
}

// ABC
template<typename T=prec>
class SmoothMoverABC {
 public:
  explicit virtual operator T() const = 0;
  virtual void operator<<(T target) = 0;
};

// Simple low-pass filter.
template<typename T=prec>
class SmoothMoverLPF : public SmoothMoverABC<T> {
 public:
  explicit SmoothMoverLPF(T initial) : s_(initial) {};
  SmoothMoverLPF(const SmoothMoverLPF&) = default;
  SmoothMoverLPF& operator=(const SmoothMoverLPF&) = default;
  SmoothMoverLPF(SmoothMoverLPF&&) = default;
  SmoothMoverLPF& operator=(SmoothMoverLPF&&) = default;
  explicit operator T() const override { return (T)s_; }
  void operator<<(T target) override {
    s_ = s_ * (1 - kLPFSlewRate) + target * kLPFSlewRate;
  }
  
 private:
  T s_;
};

// A constant acceleration, and bounded velocity.  (The acceleration
// can be lower than the constant at the end of the slew.)
template<typename T=prec>
class SmoothMoverBoundedAccel : public SmoothMoverABC<T> {
 public:
  explicit SmoothMoverBoundedAccel(T initial) : s_(initial) {};
  SmoothMoverBoundedAccel(const SmoothMoverBoundedAccel&) = default;
  SmoothMoverBoundedAccel& operator=(const SmoothMoverBoundedAccel&) = default;
  SmoothMoverBoundedAccel(SmoothMoverBoundedAccel&&) = default;
  SmoothMoverBoundedAccel& operator=(SmoothMoverBoundedAccel&&) = default;
  explicit operator T() const override { return s_; }
  void operator<<(T target) override;
  
 private:
  T s_;
  T ds_;
};

template<typename T> void
SmoothMoverBoundedAccel<T>::operator<<(T target) {
  if (target < s_) {
    ds_ -= kBoundedMaxSlewAccel;
    if (ds_ < -kBoundedMaxSlew)
      ds_ = -kBoundedMaxSlew;
    if (s_ + ds_ < target)
      ds_ = target - s_;
    s_ += ds_;
    assert(s_ >= target);
  } else if (target > s_) {
    ds_ += kBoundedMaxSlewAccel;
    if (ds_ > kBoundedMaxSlew)
      ds_ = kBoundedMaxSlew;
    if (s_ + ds_ > target)
      ds_ = s_ - target;
    s_ += ds_;
    assert(s_ <= target);
  }
}

template<typename R, typename S, typename T=prec>
class SmoothMoverCompose : public SmoothMoverABC<T> {
 public:
  explicit SmoothMoverCompose(T initial) : r_(initial), s_(initial) {};
  SmoothMoverCompose(const SmoothMoverCompose&) = default;
  SmoothMoverCompose& operator=(const SmoothMoverCompose&) = default;
  SmoothMoverCompose(SmoothMoverCompose&&) = default;
  SmoothMoverCompose& operator=(SmoothMoverCompose&&) = default;
  explicit operator T() const override { return static_cast<T>(s_); }
  void operator<<(T target) override {
    r_ << target;
    s_ << static_cast<T>(r_);
  }
  
 private:
  R r_;
  S s_;
};

// XXX After fiddling with this some, I think I came to an interesting
// conclusion.  First, perception is better tuned to seeing scaling
// effects than panning effects.  Second, it's probably better tuned
// to pixel-scale jumps than non-pixel-scale jumps.  Third, the
// "obvious" algorithms -- everything I can invent without lots of
// work -- tend to build cutoffs are divisible by 4, and hence often
// make pixel-scale jumps.  I think I should experiment within the
// x:y:width:height framework (that cv::Rect does) instead of trying
// to use x0:y0:x1:y1 like I do here.
template<typename T=prec>
class SmoothMovingRect {
 public:
  template<typename Q>
  SmoothMovingRect(cv::Rect_<Q> bounds) 
      : bounds_(bounds),
        aspect_(static_cast<T>(bounds.width) / bounds.height),
        centerX_(bounds.x + static_cast<T>(bounds.width) / 2),
        centerY_(bounds.y + static_cast<T>(bounds.height) / 2),
        size_(bounds.width * bounds.height) {};
  SmoothMovingRect(const SmoothMovingRect&) = default;
  SmoothMovingRect& operator=(const SmoothMovingRect&) = default;
  SmoothMovingRect(SmoothMovingRect&&) = default;
  SmoothMovingRect& operator=(SmoothMovingRect&&) = default;

  template<typename Q>
  cv::Rect_<Q> scale(T factor, T verticalPositioning) const;
  template<typename Q> explicit operator cv::Rect_<Q>() const {
    return scale<Q>(1, 0.5);
  }
  void operator<<(const cv::Rect_<T>&);
  
 private:
  const cv::Rect_<T> bounds_;
  const T aspect_;
  SmoothMoverCompose<SmoothMoverBoundedAccel<T>,
                     SmoothMoverLPF<T>,
                     T>
  centerX_, centerY_;
  SmoothMoverLPF<T> size_;
};

template<typename T> template<typename Q>
cv::Rect_<Q>
SmoothMovingRect<T>::scale(T factor, T verticalPositioning) const {
  T size = static_cast<T>(size_) * factor;
  T aspect = aspect_;
  T height = std::sqrt(size / aspect);
  T width = aspect * height;
  T x = static_cast<T>(centerX_) - (width / 2);
  T y = static_cast<T>(centerY_) - (height * verticalPositioning);
  // XXX Respect the bounds (we currently let the affine transform do
  // border stuff)
  return cv::Rect_<Q>(x, y, width, height);
}

template<typename T> void
SmoothMovingRect<T>::operator<<(const cv::Rect_<T>& target) {
  centerX_ << target.x + target.width / 2;
  centerY_ << target.y + target.height / 2;
  size_ << target.width * target.height;
  cout << static_cast<T>(centerX_) << ","
       << static_cast<T>(centerY_) << " @ "
       << static_cast<T>(size_) << endl;
}

template<typename T> cv::Rect_<T>
rectBound(const std::vector<cv::Rect_<T>> &rects)
{
  assert(!rects.empty());
  cv::Rect_<T> rv = rects[0];
  for (size_t i = 1; i < rects.size(); i++) {
    rv |= rects[i];
  }
  return rv;
}

int
main()
{
  cv::CascadeClassifier face_cascade;
  if (!face_cascade.load("haarcascade_frontalface_default.xml"))
    err(1, "haarcascade_frontalface_default.xml");

  int out_fd = open("/dev/video0", O_RDWR);
  if (out_fd < 0)
    err(1, "/dev/video0");
  struct v4l2_format vid_format;
  memset(&vid_format, 0, sizeof(vid_format));
  vid_format.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  if (ioctl(out_fd, VIDIOC_G_FMT, &vid_format) < 0)
    err(1, "VIDIOC_G_FMT");
  vid_format.fmt.pix.width = kOutWidth;
  vid_format.fmt.pix.height = kOutHeight;
  vid_format.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
  vid_format.fmt.pix.sizeimage = kOutHeight * kOutWidth * 3;
  vid_format.fmt.pix.field = V4L2_FIELD_NONE;
  if (ioctl(out_fd, VIDIOC_S_FMT, &vid_format) < 0)
    err(1, "VIDIOC_S_FMT");
  
  cv::VideoCapture cam(1, cv::CAP_V4L2);
  if (!cam.isOpened())
    errx(1, "open camera 1");
  cam.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
  //cam.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
  //cam.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
#define DUMP_PROP(x)                            \
  do {                                          \
      auto propVal = cam.get(cv:: x);           \
      if (propVal != -1)                        \
        cout << #x << ": " << propVal << endl;  \
  } while (0)
  DUMP_PROP(CAP_PROP_POS_MSEC);
  DUMP_PROP(CAP_PROP_POS_FRAMES);
  DUMP_PROP(CAP_PROP_POS_AVI_RATIO);
  DUMP_PROP(CAP_PROP_FRAME_WIDTH);
  DUMP_PROP(CAP_PROP_FRAME_HEIGHT);
  DUMP_PROP(CAP_PROP_FPS);
  DUMP_PROP(CAP_PROP_FOURCC);
  DUMP_PROP(CAP_PROP_FRAME_COUNT);
  DUMP_PROP(CAP_PROP_FORMAT);
  DUMP_PROP(CAP_PROP_MODE);
  DUMP_PROP(CAP_PROP_BRIGHTNESS);
  DUMP_PROP(CAP_PROP_CONTRAST);
  DUMP_PROP(CAP_PROP_SATURATION);
  DUMP_PROP(CAP_PROP_HUE);
  DUMP_PROP(CAP_PROP_GAIN);
  DUMP_PROP(CAP_PROP_EXPOSURE);
  DUMP_PROP(CAP_PROP_CONVERT_RGB);
  DUMP_PROP(CAP_PROP_WHITE_BALANCE_BLUE_U);
  DUMP_PROP(CAP_PROP_RECTIFICATION);
  DUMP_PROP(CAP_PROP_MONOCHROME);
  DUMP_PROP(CAP_PROP_SHARPNESS);
  DUMP_PROP(CAP_PROP_AUTO_EXPOSURE);
  DUMP_PROP(CAP_PROP_GAMMA);
  DUMP_PROP(CAP_PROP_TEMPERATURE);
  DUMP_PROP(CAP_PROP_TRIGGER);
  DUMP_PROP(CAP_PROP_TRIGGER_DELAY);
  DUMP_PROP(CAP_PROP_WHITE_BALANCE_RED_V);
  DUMP_PROP(CAP_PROP_ZOOM);
  DUMP_PROP(CAP_PROP_FOCUS);
  DUMP_PROP(CAP_PROP_GUID);
  DUMP_PROP(CAP_PROP_ISO_SPEED);
  DUMP_PROP(CAP_PROP_BACKLIGHT);
  DUMP_PROP(CAP_PROP_PAN);
  DUMP_PROP(CAP_PROP_TILT);
  DUMP_PROP(CAP_PROP_ROLL);
  DUMP_PROP(CAP_PROP_IRIS);
  DUMP_PROP(CAP_PROP_SETTINGS);
  DUMP_PROP(CAP_PROP_BUFFERSIZE);
  DUMP_PROP(CAP_PROP_AUTOFOCUS);
  DUMP_PROP(CAP_PROP_SAR_NUM);
  DUMP_PROP(CAP_PROP_SAR_DEN);
  DUMP_PROP(CAP_PROP_BACKEND);
  DUMP_PROP(CAP_PROP_CHANNEL);
  DUMP_PROP(CAP_PROP_AUTO_WB);
  DUMP_PROP(CAP_PROP_WB_TEMPERATURE);
  DUMP_PROP(CAP_PROP_CODEC_PIXEL_FORMAT);
  //DUMP_PROP(CAP_PROP_BITRATE);
  //DUMP_PROP(CAP_PROP_ORIENTATION_META);
  //DUMP_PROP(CAP_PROP_ORIENTATION_AUTO);
  
  int camWidth = cam.get(cv::CAP_PROP_FRAME_WIDTH);
  int camHeight = cam.get(cv::CAP_PROP_FRAME_HEIGHT);
  cv::Size outSize(kOutWidth, kOutHeight);
  cout << cam.getBackendName() << " frame size "
       << camWidth << "x" << camHeight << endl;
  uint32_t fourcc = cam.get(cv::CAP_PROP_FOURCC);
  cout << "fourcc: "
       << static_cast<char>((fourcc >>  0) & 0x7f)
       << static_cast<char>((fourcc >>  8) & 0x7f)
       << static_cast<char>((fourcc >> 16) & 0x7f)
       << static_cast<char>((fourcc >> 24) & 0x7f)
       << endl;
  
  std::string opwin("CamTrack Operator");
  std::string outwin("CamTrack Output");
  cv::namedWindow(opwin);
  cv::namedWindow(outwin);
  
  cv::Mat frame;
  cv::Mat frame_gray;
  cv::Mat output;
  std::vector<cv::Rect> faces;

  SmoothMovingRect<prec> roi(cv::Rect(0, 0, camWidth, camHeight));
  
  for (;;) {
    cam >> frame;

    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);
    face_cascade.detectMultiScale(frame_gray, faces);

    if (faces.size() > 0) {
      cv::Rect faceBounds = rectBound(faces);
      dbgRect(frame, faceBounds, cv::Scalar(0, 255, 0));
      roi << faceBounds;
    }
    dbgRect(frame, static_cast<cv::Rect>(roi), cv::Scalar(255, 0, 0));
    auto xmit = roi.scale<prec>(4, 5.0/12);
    dbgRect(frame, xmit, cv::Scalar(38, 38, 238));

    //cout << xmit << endl;
    
    try {
      cv::Point2f src[]{
                       {xmit.x, xmit.y},
                       {xmit.x, xmit.y + xmit.height},
                       {xmit.x + xmit.width, xmit.y + xmit.height}};
      cv::Point2f dst[]{
                       {0, 0},
                       {0, kOutHeight},
                       {kOutWidth, kOutHeight}};
      auto xfrm = cv::getAffineTransform(src, dst);
      cv::warpAffine(frame, output, xfrm,
                     cv::Size{kOutWidth, kOutHeight},
                     cv::INTER_CUBIC);
    } catch (cv::Exception &e) {
      // Debugging aid
      cerr << "Output " << xmit << endl;
      throw e;
    }

    cv::imshow(opwin, frame);
    cv::imshow(outwin, output);
    cv::cvtColor(output, output, cv::COLOR_BGR2RGB);
    auto written = write(out_fd, output.data, kOutHeight * kOutWidth * 3);
    if (written < 0)
      err(1, "write frame");
    
    cv::waitKey(1);
  }
}
