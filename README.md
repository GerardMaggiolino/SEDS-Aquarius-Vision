Ordinal terrain classifier for UCSD SEDS Aquarius, an autonomous lander. 

Build in progress - custom created data set unavailable on GitHub due to size, 
email gerardmaggiolino@gmail.com for access. 

Sample performance over test images: 

![](https://github.com/GerardMaggiolino/SEDS-Aquarius-Vision/blob/master/examples/test_1.png)
![](https://github.com/GerardMaggiolino/SEDS-Aquarius-Vision/blob/master/examples/test_3.png)

Currently struggling with identification of safe regions in sand: 

![](https://github.com/GerardMaggiolino/SEDS-Aquarius-Vision/blob/master/examples/test_2.png)


The data set size is currently 411 250x250 images, with no rotated duplicates. It's class imbalanced with category 3, 4, and 5 being underrepresented. 

To do: 

Increase data set size through more manual labelling (rapid with the created scripts), add weighted BCE loss to address class imbalance, expirement with lowering parameters to prevent overfitting. Rotate images at 90, 180, and 270 degrees to quadruple data set size. 

References for build: 

Cheng, J., Wang, Z., & Pollastri, G. (2008, June). A neural network approach to ordinal regression. In Neural Networks, 2008. IJCNN 2008.(IEEE World Congress on Computational Intelligence). IEEE International Joint Conference on (pp. 1279-1284). IEEE.

Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., ... & Adam, H. (2017). Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861.
