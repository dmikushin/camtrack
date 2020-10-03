// XXX Still doesn't work in Discord

// sudo modprobe v4l2loopback exclusive_caps=1
// NOT USED: v4l2loopback-ctl set-caps video/x-raw,format=UYVY,width=640,height=480 /dev/video0

#include <cassert>
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
constexpr int kOutHeight = 480;
constexpr prec kOutAspect = (prec)kOutWidth / kOutHeight;
// I put a lot of primes in here to try to keep the zoom algorithms
// from converging on small binary fractions; I figured that would
// allow the image smoothing algorithms to blur out shifts.  Sadly, it
// didn't work.
constexpr prec kLPFSlewRate = 1.0/101;
constexpr prec kBoundedMaxSlew = 43.0/41.0;
constexpr prec kBoundedMaxSlewAccel = 1.0/37;

inline void
dbgRect(cv::Mat &img __attribute__((unused)),
        cv::Rect rec __attribute__((unused)),
        const cv::Scalar &color __attribute__((unused)),
        int thickness __attribute__((unused)) = 1,
        int lineType __attribute__((unused)) = cv::LINE_8,
        int shift __attribute__((unused)) = 0) {
  //cv::rectangle(img, rec, color, thickness, lineType, shift);
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
  SmoothMovingRect(cv::Rect_<Q> initial) 
      : left_(initial.x), top_(initial.y),
        right_(initial.x + initial.width),
        bottom_(initial.y + initial.height) {};
  SmoothMovingRect(const SmoothMovingRect&) = default;
  SmoothMovingRect& operator=(const SmoothMovingRect&) = default;
  SmoothMovingRect(SmoothMovingRect&&) = default;
  SmoothMovingRect& operator=(SmoothMovingRect&&) = default;

  template<typename Q>
  explicit operator cv::Rect_<Q>() const {
    return cv::Rect_<Q>(static_cast<T>(left_),
                        static_cast<T>(top_),
                        static_cast<T>(right_) - static_cast<T>(left_),
                        static_cast<T>(bottom_) - static_cast<T>(top_));
  }
  void operator<<(const cv::Rect_<T>&);
  
 private:
  SmoothMoverCompose<SmoothMoverBoundedAccel<T>,
                     SmoothMoverLPF<T>,
                     T>
  left_, top_, right_, bottom_;
};

template<typename T> void
SmoothMovingRect<T>::operator<<(const cv::Rect_<T>& target) {
  left_ << target.x;
  right_ << target.x + target.width;
  top_ << target.y;
  bottom_ << target.y + target.height;
}

template<typename T> cv::Rect_<T>
rectBound(std::vector<cv::Rect_<T>> &rects)
{
  assert(!rects.empty());
  cv::Rect_<T> rv = rects[0];
  for (size_t i = 1; i < rects.size(); i++) {
    if (rects[i].x < rv.x)
      rv.x = rects[i].x;
    if (rects[i].x + rects[i].width < rv.x + rv.width)
      rv.width = rects[i].x + rects[i].width - rv.x;
    if (rects[i].y < rv.y)
      rv.y = rects[i].y;
    if (rects[i].y + rects[i].height < rv.y + rv.height)
      rv.height = rects[i].y + rects[i].height - rv.y;
  }
  return rv;
}

template<typename T> cv::Rect_<T>
rectAdjust(cv::Rect_<T> rect, prec aspect,
           prec scaleUp, prec scaleDown, prec scaleLeft, prec scaleRight,
           T minx, T miny, T maxx, T maxy)
{
  constexpr int convergence_iters = 32;
  
  // Scale
  T centerx = rect.x + rect.width / 2.0;
  T centery = rect.y + rect.height / 2.0;
  T left = centerx - rect.width / 2.0 * scaleLeft;
  T right = centerx + rect.width / 2.0 * scaleRight;
  T top = centery - rect.height / 2.0 * scaleUp;
  T bottom = centery + rect.height / 2.0 * scaleDown;

  int i;
  for (i = 0; i < convergence_iters; i++) {
    if (left < minx)
      left = minx;
    if (right > maxx)
      right = maxx;
    if (top < miny)
      top = miny;
    if (bottom > maxy)
      bottom = maxy;
    centerx = (left + right) / 2.0;
    centery = (top + bottom) / 2.0;
    
    // XXX We can hit a singularity here.
    prec width = right - left;
    prec height = bottom - top;
    prec newAspect = (prec)width / height;
    //cout << newAspect << " -> " << aspect << endl;
    if (newAspect < aspect - 0.01) {
      int desiredWidth = height * aspect;
      left = centerx - desiredWidth / 2;
      right = left + desiredWidth;
      continue;
    }
    if (newAspect > aspect + 0.01) {
      int desiredHeight = width / aspect;
      top = centery - desiredHeight / 2;
      bottom = top + desiredHeight;
      continue;
    }
    break;
  }

  static bool in_fallback;
  static cv::Rect_<T> last_value(minx, miny, maxx-1, maxy-1);
  if (i == convergence_iters || (right <= left) || (bottom <= top)) {
    // Failed to converge.  Not really that surprising; the above
    // algorithm needs a lot of work.  Return a default.
    // In my tests so far, this will go in and out of fallback
    // rapidly for several frames, and then resolve long-term.    
    if (!in_fallback) {
      cerr << "CONVERGENCE FAILURE" << endl;
      in_fallback = true;
    }
    return last_value;
  }

  if (in_fallback) {
    cerr << "Convergence resolved" << endl;
    in_fallback = false;
  }
  last_value = cv::Rect_<T>(left, top, right - left, bottom - top);
  return last_value;
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
  cout << "opened using " << cam.getBackendName() << endl;
  int camWidth = cam.get(cv::CAP_PROP_FRAME_WIDTH);
  int camHeight = cam.get(cv::CAP_PROP_FRAME_HEIGHT);
  cv::Size outSize(kOutWidth, kOutHeight);
  
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
    // XXX The ROI can be backwards, and if so, this doesn't draw it.
    dbgRect(frame, static_cast<cv::Rect>(roi), cv::Scalar(255, 0, 0));
    auto xmit = rectAdjust<prec>(static_cast<cv::Rect>(roi), kOutAspect,
                                 // This should have scaleBottom ==
                                 // 2*scaleTop to have the eyes on the
                                 // rule-of-thirds boundary.  (It also
                                 // works well with my beard.)
                                 1.5, 3, 2, 2,
                                 0, 0, camWidth, camHeight);
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

    //cv::imshow(opwin, frame);
    //cv::imshow(outwin, output);
    cv::cvtColor(output, output, cv::COLOR_BGR2RGB);
    auto written = write(out_fd, output.data, kOutHeight * kOutWidth * 3);
    if (written < 0)
      err(1, "write frame");
    
    cv::waitKey(1);
  }
}
