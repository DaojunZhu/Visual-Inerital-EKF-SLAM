
#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc,char** argv)
{
    string save_path = 
        "/home/daojun/myrosprojects_ws/src/VIEKFSLAM2/aruco_markers";

    int marker_num = 15;
    
    Ptr<aruco::Dictionary> dictionary = 
        aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    
    for(int i = 15; i < 20; ++i){
        Mat markerImage;
        aruco::drawMarker(dictionary,i,400,markerImage,1);
        string marker_name = save_path + "/marker" + to_string(i) + ".png";
        imwrite(marker_name,markerImage);
    }

    return 0;
}