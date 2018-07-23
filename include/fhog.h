#ifndef SX_FHOG_FHOG_H
#define SX_FHOG_FHOG_H

#include <opencv2\core.hpp>

#define FHOG_OK             0
#define FHOG_ERR_FEATUREMAP 1
#define FHOG_ERR_NORM       2
#define FHOG_ERR_PCA        3

#define NUM_SECTOR 9

class FHOG{
public:
    FHOG():imageSize(cv::Size(0,0)),
           cellSize(0),
           sz(cv::Size(0,0)),
           numFeatures(0){};
    ~FHOG(){};
    
    /***********************************************************************
     * Function: int FHOG::static_Init(cv::Size _sz, int cellSize);
     * Parameter:
     *      cv::Size    sz          Input Image Size
     *      int         cellSize    HOG cell size
     * Output: 
     *      int         status      FHOG
     * Description:
     *      This function is used to pre alloc data used by FHOG.
     *      This function will be automatically called once the size of src
     *   is differnt to previous size. This function can be called by users
     *   if the time must be constant.
     ***********************************************************************/
    int static_Init(cv::Size _sz, int _cellSize);

    int compute(const cv::Mat &src, cv::Mat &dst, int _cellSize, float thres);

private:
    int getFeatureMaps(const cv::Mat &src);
    int normalizeAndTruncate(float thres);
    int PCAFeatureMaps();

    cv::Size imageSize;
    int cellSize;
    cv::Size sz;
    int numFeatures;
    cv::Mat map; // sz.y*sz.x X numFeatures 
    cv::Mat originalFeature; // (sz.y+2)*(sz.x+2) X numFeatures X 1
    cv::Mat normalizedFeature; // sz.y*sz.x X numFeatures X 1

// Used by getFeatureMaps()
    cv::Mat alpha; // (sz.y+2)*cellSize X (sz.x+2)*cellSize X 2
    cv::Mat r; // (sz.y+2)*cellSize X (sz.x+2)*cellSize X 1
    
    cv::Mat nearest; // CellSize
    cv::Mat w; // 2*CellSize

    cv::Mat dx; // img.rows x img.cols
    cv::Mat dy; // img.rows x img.cols
    cv::Mat dx_ori; // (img.rows+2) x (img.cols+2)
    cv::Mat dy_ori; // (img.rows+2) x (img.cols+2)
    cv::Mat imagePadded; // (img.rows+2) x (img.cols+2)

// Used by normalizeAndTruncate()
    cv::Mat partOfNorm; // (sz.y+2)*(sz.x+2) (float)
    
// Used by PCAFeatureMaps()

};

#endif
