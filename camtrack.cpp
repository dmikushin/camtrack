// sudo modprobe v4l2loopback exclusive_caps=1
// v4l2loopback-ctl set-caps video/x-raw,format=UYVY,width=640,height=480 /dev/video0

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

constexpr int kOutWidth = 640;
constexpr int kOutHeight = 480;
constexpr float kOutAspect = (float)kOutWidth / kOutHeight;
constexpr float kLPFSlewRate = 0.01;
constexpr float kBoundedMaxSlew = 1;
constexpr float kBoundedMaxSlewAccel = 0.05;

// ABC
template<typename T=float>
class SmoothMoverABC {
 public:
  explicit virtual operator T() const = 0;
  virtual void operator<<(T target) = 0;
};


// Simple low-pass filter.
template<typename T=float>
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
template<typename T=float>
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

template<typename R, typename S, typename T=float>
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

template<typename T=float>
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

cv::Rect
rectAdjust(cv::Rect rect, float aspect,
           float scaleUp, float scaleDown, float scaleLeft, float scaleRight,
           int minx, int miny, int maxx, int maxy)
{
  constexpr int convergence_iters = 32;
  
  // Scale
  float centerx = rect.x + rect.width / 2.0;
  float centery = rect.y + rect.height / 2.0;
  float left = centerx - rect.width / 2.0 * scaleLeft;
  float right = centerx + rect.width / 2.0 * scaleRight;
  float top = centery - rect.height / 2.0 * scaleUp;
  float bottom = centery + rect.height / 2.0 * scaleDown;

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
    int width = right - left;
    int height = bottom - top;
    float newAspect = (float)width / height;
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
  static cv::Rect last_value(minx, miny, maxx-1, maxy-1);
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
  last_value = cv::Rect(left, top, right - left, bottom - top);
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

  SmoothMovingRect<float> roi(cv::Rect(0, 0, camWidth, camHeight));
  
  for (;;) {
    cam >> frame;

    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);
    face_cascade.detectMultiScale(frame_gray, faces);

    if (faces.size() > 0) {
      cv::Rect faceBounds = rectBound(faces);
      cv::rectangle(frame, faceBounds, cv::Scalar(0, 255, 0));
      roi << faceBounds;
    }
    // XXX The ROI can be backwards, and if so, this doesn't draw it.
    cv::rectangle(frame, static_cast<cv::Rect>(roi), cv::Scalar(255, 0, 0));
    auto xmit = rectAdjust(static_cast<cv::Rect>(roi), kOutAspect,
                           // This should have scaleBottom ==
                           // 2*scaleTop to have the eyes on the
                           // rule-of-thirds boundary.  (It also
                           // works well with my beard.)
                           1.5, 3, 2, 2,
                           0, 0, camWidth, camHeight);
    cv::rectangle(frame, xmit, cv::Scalar(38, 38, 238));

    try {
      auto outCols = frame.colRange(xmit.x, xmit.x + xmit.width);
      auto outColsRows = outCols.rowRange(xmit.y, xmit.y + xmit.height);
      cv::resize(outColsRows, output, outSize, 0, 0, cv::INTER_AREA);
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
