# InnerEye-classifiers
Color of daily objects says a lot about them. However, upon applying image filters, which are often available on social media, objects in an image undergo a color transformation. The objects in such an image often have color that confuses us or conveys other meaning. Therefore, applying an image filter is a kind of image editing.

<!--<p align="center">
  <img width="128" height="192" src="https://github.com/greenboal/InnerEye-classifiers/blob/master/sample_images/Me.jpg">
  <br>
  Figure: An example of unedited image.
</p>-->

<!--<p align="center">
  <img width="128" height="192" src="https://github.com/greenboal/InnerEye-classifiers/blob/master/sample_images/Me_nashville.jpg">
  <br>
  Figure: Edited image due to application of Nashville filter.
</p>-->

<!--<p align="center">
  <img width="128" height="192" src="https://github.com/greenboal/InnerEye-classifiers/blob/master/sample_images/Me_xpro2.jpg">
  <br>
  Figure: Edited image due to application of XPro2 filter.
</p>-->


This classifier, based on the survey result of the ongoing [InnerEye](http://inner-eye.herokuapp.com/) project, can distinguish such edited and unedited images. InnerEye project aims at understanding the credibility of social interaction in the presence of both edited and unedited images on the social media platforms.


In this classifier, the challenge is to recognize how the color style of an image is different from it's unedited (images on which no filters has been applied) counterpart. However, under different illumination condition this color style changes. Assuming there is an invariant relationship among the colors for each of the image filters, this classifier has been built. To compile the dataset required, images are sampled from the [Google Landmarks dataset](https://www.kaggle.com/google/google-landmarks-dataset) first and then different image filters are applied to the images. Both the unedited and the edited counterparts of those are present in the dataset.


Because the color style of an image has to be understood, the content and the color of an image has to be separated. We need to work with only the style of the input image. The style and content separating mechanism in the classifier is based on the autoencoder architecture in [MUNIT](http://openaccess.thecvf.com/content_ECCV_2018/html/Xun_Huang_Multimodal_Unsupervised_Image-to-image_ECCV_2018_paper.html) InnerEye classifier is then cotrained on the images (image reconstruction) and the edited image labels (unedited or edited). The classifier is multi-targeted to improve accuracy- therefore, the class target labels are {'unedited', '_1977', 'aden', 'brannan', 'brooklyn', 'clarendon', 'earlybird', 'gingham', 'hudson', 'inkwell', 'kelvin', 'lark', 'lofi', 'maven', 'mayfair', 'moon', 'nashville', 'perpetua', 'reyes', 'rise', 'slumber', 'stinson', 'toaster', 'valencia', 'walden', 'willow', 'xpro2'}.


<p align="center">
  <img width="522" height="208" src="https://github.com/greenboal/InnerEye-classifiers/blob/master/architectures/InnerEye Architecture.png">
  <br>
  Figure: Architecture of the classifier.
</p>


A classifier that does not separate style and content and classifies on the style does not converge.

<p align="center">
  <img width="422" height="336" src="https://github.com/greenboal/InnerEye-classifiers/blob/master/Sequential/Seq-30-loss.png">
  <br>
  Figure: Loss of the sequential classifier.
</p>


However, a classifier (our contribution) that separates style and content and classifies on the style converges.

<p align="center">
  <img width="422" height="336" src="https://github.com/greenboal/InnerEye-classifiers/blob/master/analytical_classifier/A-30-loss.png">
  <br>
  Figure: Loss of the analytical classifier.
</p>

<!--We made the training dataset smaller- reducing the number of training images to one seventh, and then trained both the network with hyperparameters unchanged. Surprisingly, our network (training is still underway) outperformed the sequential one by a wide margin. Also, our network has yet not overfitted unlike to its counterpart.-->

<!--<p align="center">
  <img width="422" height="326" src="https://github.com/greenboal/InnerEye-classifiers/blob/master/sample_images/on_lighter_dataset.png">
  <br>
  Figure: Comaparison of both the neural networks.
</p>-->


The author of the classifier is available at 1018052026@grad.cse.buet.ac.bd
