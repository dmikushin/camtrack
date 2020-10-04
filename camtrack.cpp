#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>

#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <err.h>
#include <fcntl.h>
#include <unistd.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudawarping.hpp>
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
  SmoothMovingRect(cv::Rect_<Q> bounds, T aspect)
      : bounds_(bounds),
        aspect_(aspect),
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

#if 0
  cout << width << "x"
       << height << "+"
       << x << "+"
       << y << " / "
       << bounds_.width << "x"
       << bounds_.height << "+"
       << bounds_.x << "+"
       << bounds_.y;
#endif
  
  // Shrink it into bounds, while maintaining the aspect ratio.  We
  // always shrink from all edges, to maintain the center.
  if (x < 0) {
    T delta = -x;
    width -= 2 * delta;
    height -= 2 * delta / aspect;
    x = 0;
  }
  if (y < 0) {
    T delta = -y;
    height -= 2 * delta;
    width -= 2 * delta * aspect;
    y = 0;
  }
  T right = x + width;
  T bounds_right = bounds_.x + bounds_.width;
  if (right > bounds_right) {
    T delta = right - bounds_right;
    width -= 2 * delta;
    height -= 2 * delta / aspect;
    x += delta;
    y += delta / aspect;
  }
  T bottom = y + height;
  T bounds_bottom = bounds_.y + bounds_.height;
  if (bottom > bounds_bottom) {
    T delta = bottom - bounds_bottom;
    height -= 2 * delta;
    width -= 2 * delta * aspect;
    y += delta;
    x += delta * aspect;
  }

#if 0
  cout << " -> "
       << width << "x"
       << height << "+"
       << x << "+"
       << y << endl;
#endif
  
  return cv::Rect_<Q>(x, y, width, height);
}

template<typename T> void
SmoothMovingRect<T>::operator<<(const cv::Rect_<T>& target) {
  centerX_ << target.x + target.width / 2;
  centerY_ << target.y + target.height / 2;
  size_ << target.width * target.height;
#if 0
  cout << static_cast<T>(centerX_) << ","
       << static_cast<T>(centerY_) << " @ "
       << static_cast<T>(size_) << endl;
#endif
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

class RawV4L2CudaSource : public cv::cudacodec::RawVideoSource {
 public:
  explicit RawV4L2CudaSource(const char* filename);
  RawV4L2CudaSource(const RawV4L2CudaSource&) = delete;
  RawV4L2CudaSource& operator=(const RawV4L2CudaSource&) = delete;
  RawV4L2CudaSource(RawV4L2CudaSource&&) = delete;
  RawV4L2CudaSource& operator=(RawV4L2CudaSource&&) = delete;

  virtual ~RawV4L2CudaSource() override;
  virtual cv::cudacodec::FormatInfo format() const override;
  virtual bool getNextPacket(unsigned char **data, size_t *size) override;

  int width() const { return width_; }
  int height() const { return height_; }
  
 private:
  int fd_;
  struct v4l2_format vid_format_;
  //cv::cuda::HostMem buf_(cv::cuda::HostMem::AllocType::WRITE_COMBINED);
  std::vector<char> buf_;
  int width_ = 0, height_ = 0;
};

RawV4L2CudaSource::RawV4L2CudaSource(const char* filename) {
  fd_ = open(filename, O_RDWR);
  if (fd_ < 0)
    err(1, filename);

  memset(&vid_format_, 0, sizeof(vid_format_));
  vid_format_.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (ioctl(fd_, VIDIOC_G_FMT, &vid_format_) < 0)
    err(1, "%s: initial VIDIOC_G_FMT", filename);
  vid_format_.fmt.pix.width = 1920;
  vid_format_.fmt.pix.height = 1080;
  vid_format_.fmt.pix.pixelformat = V4L2_PIX_FMT_JPEG;
  vid_format_.fmt.pix.field = V4L2_FIELD_NONE;
  vid_format_.fmt.pix.bytesperline = 0;
  vid_format_.fmt.pix.sizeimage = 0;
  vid_format_.fmt.pix.priv = 0;
  if (ioctl(fd_, VIDIOC_S_FMT, &vid_format_) < 0)
    err(1, "%s: VIDIOC_S_FMT", filename);
  // Record what we got
  if (ioctl(fd_, VIDIOC_G_FMT, &vid_format_) < 0)
    err(1, "%s: followup VIDIOC_G_FMT", filename);

  //buf_.create(1, vid_format_.fmt.pix.sizeimage, CV_8U);
  buf_.resize(vid_format_.fmt.pix.sizeimage);
  width_ = vid_format_.fmt.pix.width;
  height_ = vid_format_.fmt.pix.height;  
}
  
RawV4L2CudaSource::~RawV4L2CudaSource() {
  if (close(fd_) < 0)
    err(1, "close camera");
}

cv::cudacodec::FormatInfo
RawV4L2CudaSource::format() const {
  cv::cudacodec::FormatInfo rv;
  memset(&rv, 0, sizeof(rv));
  assert(vid_format_.fmt.pix.pixelformat == V4L2_PIX_FMT_JPEG);
  // The only format that the decoder supports is 8-bit YUV420.
  // (Actually, it uses NV12 internally.)  FIXME Verify this (cf
  // V4L2_JPEG_CHROMA_SUBSAMPLING_420 etc)
  rv.chromaFormat = cv::cudacodec::ChromaFormat::YUV420;
  rv.codec = cv::cudacodec::Codec::JPEG;
  rv.width = vid_format_.fmt.pix.width;
  rv.height = vid_format_.fmt.pix.height;
  return rv;
}
  
bool
RawV4L2CudaSource::getNextPacket(unsigned char **data, size_t *size) {
  *data = static_cast<unsigned char*>(static_cast<void*>(buf_.data()));
  auto read_result = read(fd_, *data, vid_format_.fmt.pix.sizeimage);
  if (read_result < 0)
    err(1, "read frame");
  *size = read_result;
  return (read_result != 0);
}

int
main()
{
  cout << "CUDA devices: " << cv::cuda::getCudaEnabledDeviceCount() << endl;
  cv::cuda::setDevice(0);
  cv::cuda::printShortCudaDeviceInfo(0);
  
  auto face_cascade = cv::cuda::CascadeClassifier::create(
      "haarcascade_cuda.xml");

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
  // Chrome only supports:
  // V4L2_PIX_FMT_YUV420, V4L2_PIX_FMT_Y16, V4L2_PIX_FMT_Z16, 
  // V4L2_PIX_FMT_INVZ, V4L2_PIX_FMT_YUYV, V4L2_PIX_FMT_RGB24, 
  // V4L2_PIX_FMT_MJPEG, V4L2_PIX_FMT_JPEG
  // Discord doesn't work with RGB24, but does with YUV420.
  vid_format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUV420;
  vid_format.fmt.pix.sizeimage = kOutHeight * kOutWidth * 3 / 2;
  vid_format.fmt.pix.field = V4L2_FIELD_NONE;
  if (ioctl(out_fd, VIDIOC_S_FMT, &vid_format) < 0)
    err(1, "VIDIOC_S_FMT");

  auto cam_source = cv::makePtr<RawV4L2CudaSource>("/dev/video1");
  auto cam = cv::cudacodec::createVideoReader(cam_source);
  
  std::string opwin("CamTrack Operator");
  std::string outwin("CamTrack Output");
  cv::namedWindow(opwin, cv::WINDOW_OPENGL);
  cv::namedWindow(outwin, cv::WINDOW_OPENGL);

  cv::cuda::Stream background;
  cv::cuda::Stream operator_stream;
  
  cv::cuda::GpuMat input;
  cv::cuda::GpuMat input_gray;
  cv::Mat operator_display;
  cv::cuda::GpuMat faces_gpu;
  std::vector<cv::Rect> faces_cpu;
  cv::cuda::GpuMat output;
  cv::Mat output_cpu;

  SmoothMovingRect<prec> roi(cv::Rect(
      0, 0, cam_source->width(), cam_source->height()), kOutAspect);

  int interval_frames = 0;
  auto interval_start = std::chrono::steady_clock::now();

  // cv::cudacodec outputs its frames in BGRA, it seems.
  // (See videoDecPostProcessFrame)
  while(cam->nextFrame(input)) {
    cv::cuda::cvtColor(input, input_gray, cv::COLOR_BGRA2GRAY, 0, background);
    cv::cuda::equalizeHist(input_gray, input_gray, background);
    face_cascade->detectMultiScale(input_gray, faces_gpu, background);
    input.download(operator_display, operator_stream);

    background.waitForCompletion();
    face_cascade->convert(faces_gpu, faces_cpu);

    operator_stream.waitForCompletion();
    cv::Rect faceBounds;
    if (faces_cpu.size() > 0) {
      faceBounds = rectBound(faces);
      roi << faceBounds;
    }
    auto xmit = roi.scale<prec>(7, 5.0/12);

    cout << xmit << endl;
    
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
      cv::cuda::warpAffine(input, output, xfrm,
                           cv::Size{kOutWidth, kOutHeight},
                           cv::INTER_LINEAR,
                           cv::BORDER_CONSTANT,
                           cv::Scalar(),
                           background);
    } catch (cv::Exception &e) {
      // Debugging aid
      cerr << "Output " << xmit << endl;
      throw e;
    }

#if 0
    // Now that the output has been drawn, we can draw on the operator
    // window.
    if (faces.size() > 0) {
      dbgRect(frame, faceBounds, cv::Scalar(0, 255, 0));
    }
    dbgRect(frame, static_cast<cv::Rect>(roi), cv::Scalar(255, 0, 0));
    dbgRect(frame, xmit, cv::Scalar(38, 38, 238));
#endif
    
    cv::imshow(opwin, operator_display);
    background.waitForCompletion();  // Wait for the affine transform
    cv::imshow(outwin, output);
    cv::cuda::cvtColor(output, output, cv::COLOR_BGRA2YUV_I420, 0, background);
    output.download(output_cpu, background);
    background.waitForCompletion();  // Wait for the download
    auto written = write(out_fd, output_cpu.data,
                         kOutHeight * kOutWidth * 3 / 2);
    if (written < 0)
      err(1, "write frame");
    
    cv::waitKey(1);

    interval_frames++;
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> interval_duration = now - interval_start;
    if (interval_duration.count() >= 1) {
      int fps = interval_frames / interval_duration.count();
      cout << "FPS: " << fps << endl;
      interval_frames = 0;
      interval_start = now;
    }
  }
}

// Local Variables:
// compile-command: "g++ -Werror -Wall -Wextra -O2 -g -I/usr/include/opencv4 camtrack.cpp -lopencv_videoio -lopencv_highgui -lopencv_imgproc -l opencv_objdetect -lopencv_core -o camtrack"
// End:
