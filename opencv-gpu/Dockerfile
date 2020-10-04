FROM debian:bullseye AS patch
WORKDIR /build
# Download the prerequisites to build the Debian-packaged OpenCV.  Cache
# that as its own layer for ease of iteration on future steps.
RUN sed -i -e 's/main/main contrib non-free/' /etc/apt/sources.list &&  \
    sed -e 's:^deb :deb-src :'                                          \
        /etc/apt/sources.list > /etc/apt/sources.list.d/deb-src.list && \
    apt-get update && apt-get -y install --no-install-recommends        \
        build-essential                                                 \
        devscripts                                                      \
    && apt-get source opencv                                            \
    && apt-get build-dep -y opencv                                      \
    && rm -rf /var/lib/apt/lists/*
# Download the prerequisites for our changes.
RUN apt-get update && apt-get -y install --no-install-recommends        \
    libnvcuvid1                                                         \
    libnvidia-encode1                                                   \
    libqt5opengl5-dev                                                   \
    nvidia-cuda-dev                                                     \
    nvidia-cuda-toolkit                                                 \
    nvidia-driver-libs                                                  \
    && rm -rf /var/lib/apt/lists/*
# I do the MJPEG decoding in the GPU, which requires the NVCUVID
# library.  While I'm at it, I also get the encoding library.
#
# The header files for the codec are Nvidia-proprietary, and can't be
# redistributed; they're taken from the Nvidia Video Codec SDK (I'm
# using 10.0.26, not that it matters).  The libraries themselves
# aren't part of the SDK; they're part of the drivers.  (Why is this
# SDK licensed the way it is?)
COPY cuviddec.h nvcuvid.h nvEncodeAPI.h /usr/include/
# Patch 2554 re-enables the CUDA-based Haar cascade classifier.  It
# was disabled because it was "obsolete", but never replaced.  That
# patch (now at TOT) re-enables it and fixes it.  We also explicitly
# turn on NVCUVID detection: it was disabled because of
# https://github.com/opencv/opencv/issues/14850 which was since
# closed, but NVCUVID autodetection was never re-enabled (at least as
# of 4.2).
#
# We also have the additions to debian/control and the .install files
# for the two new libraries this creates.  (It also adds new
# functionality to existing libraries, like adding Qt and OpenGL
# support to libopencv-highgui).
COPY 2554.patch control.gpu                                             \
     libopencv-cuda4.2.install libopencv-cuda-dev.install               \
     libopencv-cvv4.2.install libopencv-cvv-dev.install                 \
     /build/
# The Milky icon set was removed from the dfsg build for copyright
# reasons.  It's only used by the Qt GUI, and the Debian package uses
# GTK3.  But the GTK3 frontend doesn't support OpenGL.  (The
# GTK2+GtkGlExt frontend supports OpenGL, but the GTK3 frontend
# doesn't.)  So we use the Qt GUI, and add the Milky icon set.
ADD Milky.tar.gz /build/opencv-4.2.0+dfsg
# Make our patches.  This mostly is only changing the build flags we
# use, but we also apply patch 2554 (described above).
RUN cd opencv-4.2.0+dfsg/debian &&                                      \
    dch --local=+gpu "Add OpenGL and CUDA support" &&                   \
    cat /build/control.gpu >> control &&                                \
    cp /build/libopencv-*.install . &&                                  \
    sed -i -e 's/-DWITH_CUDA=OFF/-DWITH_CUDA=ON -DWITH_NVCUVID=ON -DWITH_OPENGL=ON/' \
           -e '/CMAKE_ARCH_FLAGS += -DCPU_BASELINE/d'                   \
           -e 's:\(CMAKE_FLAGS =\):\1-DCUDA_HOST_COMPILER=/usr/bin/gcc-8:' \
           -e 's/-DWITH_QT=OFF/-DWITH_QT=ON/'                           \
           -e 's/-DWITH_GTK=ON/-DWITH_GTK=OFF/'                         \
           rules &&                                                     \
    patch -d ../contrib -p1 < /build/2554.patch
# Build the binary packages.  Use -i. to prevent debuild from trying
# to check the contents of patch 2554.  Use -nc so that the user can
# easily get access to the contents of the build when this is done,
# in addition to the .deb files.
RUN cd opencv-4.2.0+dfsg &&                                             \
    debuild -i. -uc -us -b -nc

# TODO: Rebuilds might need to sed -i -e '/-static\/lib/s/$/ || true/'
# debian/rules since the override_dh_install step deletes that.  But
# there's other problems with that way too.
# TODO: No idea why it's not finding the lapack library.  You'd think
# that an image processing library might really benefit from a good
# BLAS/LAPACK.
