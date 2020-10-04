This is how I build a version of OpenCV for Debian that supports CUDA and OpenGL.

You’ll need to download the [Nvidia Video Codec SDK](https://developer.nvidia.com/nvidia-video-codec-sdk/download) and copy the header files from its `include` directory to this directory.  You don’t need to worry about matching versions; the library is designed to work even if the headers aren’t matched to the library version.

You’ll also need to create `Milky.tar.gz` from the OpenCV distribution.  For copyright reasons, it’s not included here.  FIXME Write directions about how to do that.

FIXME Write directions about how to build this, and to get the built `.deb` files onto your local system to install with `dpkg -i`.
