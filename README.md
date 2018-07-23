# FHOG
Felzenszwalb HOG(FHOG) Feature Extractor for C++.

FHOG is widely used in object tracking and object detection. @joaofaro offers C++ implementation along with his KCF tracker. This implementation uses OpenCV 1 data structure. Besides, feature map memory is not automatically released.

This repository rewrite the FHOG code in C++ with OpenCV3.

## Requirement
1. OpenCV 3+(OpenCV2 should work, but not tested)

## Usage

```C++
#include "fhog.h"

...
    FHOG fhogDescripter;
    // This can be called before your caculation.
    // If this function is called before actual calculation,
    // many internel Mats will be pre-allocated.
    // When you are calculating fixed size FHOG, this function 
    // is useful. If you don't want to call this function,
    // it is still OK because compute function will still call
    // it automatically.
    fhogDescripter.static_Init(img.size(), 4);
    cv::Mat feat;
    // Compute the FHOG feature
    fhogDescripter.compute(img, feat, 4, 0.2);
...

```

If you only need to use fhog, copy ```include/fhog.h``` and ```src/fhog.cpp``` to your workspace. 

WARNING: ```include/fhog1.hpp``` and ```src/fhog2.cpp``` are original FHOG used to test performance.

## Performance
According to our test, our implementation is about 26% faster than original work(fixed size);

Time Consumptioin:

| Original |  New  |
|----------|-------|
| 4.96s    | 3.66s |

\* Test Condition: 640x512 Gray Image, 500 times
