Real Time Object Recognition in OpenCV using SURF Features

1. Extract SURF descriptors for the object of interest - the same is for every frame captured from the webcam
2. Match the descriptors from every frame in the webcam to the descriptors of the desired object
3. To get a bounding box around a detected object,  using the good matches, find a homography that transforms points from the object image to the video frame
4. Using this homography, transform the 4 corners of the object image. Consider these 4 transformed points as vertices and draw a box in the video frame
