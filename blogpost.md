# A deep dive into color space equivariant networks

**Authors:** *S.R. Abbring, H.C. van den Bos, R. den Braber, A.J. van Breda, D. Zegveld*

**Supervised by**: *Alejandro García Castellanos, a.garciacastellanos@uva.nl*

---

In this blog post, we discuss, analyze, and extend upon the findings of the paper titled *Color Equivariant Convolutional Networks* [[5]](#main). The paper introduces Color Equivariant Convolutions (CEConvs) and their application to color equivariance in convolutional neural networks.
The objectives of this blog post are to:

1. Discuss the methods introduced in the paper
1. Verify the authors' claims
1. Extend the notion of color equivariance to other dimensions beyond hue by leveraging different color spaces than RGB

---

## Introduction

Color is a crucial feature in how people identify and recognize objects. For example, a study by [[8]](#bird) found that color facilitates expert bird watchers in faster and more accurate recognition of bird species.

Although color invariance has been achieved in various research areas, such as in facial recognition to mitigate the influence of lighting conditions [[7]](#color_invariance), some classification problems benefit from color equivariance rather than invariance.

The Color Equivariant Convolutions (CEConvs) introduced in [[5]](#main) achieve equivariance to discrete hue shifts. Hue is represented in RGB space as a 3D rotation around the [1, 1, 1] axis. This approach exploits group theory to build equivariance into convolutional neural networks.


## Recap on Group Equivariant Convolutions

Deep Convolutional Neural Networks have been proven to be highly effective for image classification [[10]](#DCNN). Empirical evidence shows the importance of depth for good performance and convolution provides translation equivariance through weight sharing.
As a result, the convolutional layers in a deep network are translation equivariant: the output shifts relative to shifts in the input. 

Equivariance can be extended to larger groups, including rotation. This generalization is achieved through Group Convolutional Neural Networks (G-CNN). A CNN layer is equivariant to a group if transformations in the input space are related to transformations in the output space. Mathematically, this is defined as:

$$\begin{align*}
\Phi(T_g x) = T'_g \Phi(x) \qquad \forall g \in G
\tag{1}
\end{align*}$$

where $T_g$ and $T'_g$ can be equivalent.
We utilize the equations from [[1]](#group_convs) to show that G-CNNs are equivariant. Instead of shifting a filter, correlation in the first layer can be described more generally by replacing it with a group convolution:

$$\begin{align*}
[f \star \psi](g) = \sum_{y \in \mathbb{Z}^2}\sum_{k} f_k(y)\, \psi_{k}(g^{-1}y)
\tag{2}
\end{align*}$$

Since the feature map $f \star \psi$ is a function on G, the filters are functions on G for all layers after the first. The correlation then becomes:

$$\begin{align*}
[f \star \psi](g) = \sum_{h \in G}\sum_{k} f_k(h)\, \psi_{k}(g^{-1}h)
\tag{3}
\end{align*}$$

We define the left regular representation, whereby the group is acting on the transitive input space of the function $f: X \rightarrow Y$:

$$\begin{align*}
[L_g f](x) = [f \circ g^{-1}](x) = f(g^{-1}x)
\tag{4}
\end{align*}$$

Using this representation and the substitution $h \rightarrow uh$, the equivariance of the correlation can be derived such that a translation followed by a correlation is equivalent to a correlation followed by a translation:

$$\begin{align*}
[[L_u f] \star \psi](g)
  &= \sum_{h \in G}\sum_k f_k(u^{-1}h)\,\psi(g^{-1}h) \\
  &= \sum_{h \in G}\sum_k f(h)\,\psi(g^{-1}uh) \\
  &= \sum_{h \in G}\sum_k f(h)\,\psi((u^{-1}g)^{-1}h) \\
  &= [L_u[f \star \psi]](g)
\tag{5}
\end{align*}$$

## Color Equivariance

The original paper exploits the concept of group equivariant convolutions to achieve color equivariance, defined as equivariance to hue shifts. In the HSV (Hue-Saturation-Value) color space, hue is represented as an angle.

This definition is extended to group theory, by defining the group $H_n$ as a subgroup of the $SO(3)$ group. Specifically, $H_n$ consists of multiples of $\frac{360}{n}$-degree rotations along the [1, 1, 1] axis in RGB space.

This leads to the following parameterization of $H_n$, with $n$ identifying the total number of discrete rotations, $k$ encoding a specific rotation, $a = \frac{1}{3} - \frac{1}{3}\cos\!\left(\frac{2k\pi}{n}\right)$ and $b = \frac{1}{3}\sqrt{2}\sin\!\left(\frac{2k\pi}{n}\right)$:

$$\begin{align*}
H_n =
\begin{bmatrix}
\cos\!\left(\tfrac{2k\pi}{n}\right) + a & a - b & a + b \\
a + b & \cos\!\left(\tfrac{2k\pi}{n}\right) + a & a - b \\
a - b & a + b & \cos\!\left(\tfrac{2k\pi}{n}\right) + a
\end{bmatrix}
\tag{6}
\end{align*}$$

The group of discrete hue shifts is combined with the group of discrete 2D translations into the group $G = \mathbb{Z}^2 \times H_n$. The Color Equivariant Convolution (CEConv) in the first layer is defined as:

$$\begin{align*}
[f \star \psi^i](x, k) = \sum_{y \in \mathbb{Z}^2}\sum_{c=1}^{C^l} f_c(y) \cdot H_n(k)\,\psi_c^i(y - x)
\tag{7}
\end{align*}$$

However, a small error is present here as the sum $\sum_{c=1}^{C^l}$ indicates that $f_c(y)$ and $\psi_c^i(y - x)$ are scalar values. This interpretation is inconsistent given the dot product and matrix operation involved.
Therefore the correct formula should be:

$$\begin{align*}
[f \star \psi^i](x, k) = \sum_{y \in \mathbb{Z}^2} f(y) \cdot H_n(k)\,\psi^i(y - x)
\tag{8}
\end{align*}$$

However, this change does not impact the derivation of the equivariance of the CEConv layer, as given in the original paper [[5]](#main).

For the hidden layers, the feature map $[f \star \psi]$ is a function on $G$ parameterized by $x$ and $k$. The CEConv hidden layers formula then becomes:

$$\begin{align*}
[f \star \psi^i](x, k) = \sum_{y \in \mathbb{Z}^2}\sum_{r=1}^{n} f(y,r) \cdot \psi^i\!\left(y - x,\; (r-k) \bmod n\right)
\tag{9}
\end{align*}$$

## Reproduction of Experiments

The reproduction of (a selection of) the experiments is primarily achieved through the code provided along with the original paper. However, the code for reproducing the plots was not included and had to be recreated.

### When is color equivariance useful?

First, the experiments that show the importance of color equivariance are reproduced. This mainly includes exploring various datasets, starting with a color-imbalanced dataset and followed by the investigation of color selectivity.

#### Color Imbalance

To verify that color equivariance can share shape information across classes, we reproduced the long-tailed ColorMNIST experiment. In this experiment, a 30-way classification is performed on a power law distributed dataset where some classes have significantly fewer samples than others.

Two models were tested; the Z2CNN [[1]](#group_convs), a vanilla 7-layer ResNet-18 CNN model, consisting of 25,990 trainable parameters, and the CECNN model, a corresponding 7-layer ResNet-18 CNN that incorporates color equivariance with parameter sharing.

<div align="center">
  <img src="blogpost_imgs/Longtailed.png" alt="Longtailed dataset results" width="600">

  *Figure 1: Classification performance of a normal CNN (Z2CNN) and the color equivariant CNN (CECNN) on a long-tailed, unequally distributed dataset, illustrating the impact of weight sharing in the CECNN.*
</div>

The x-axis of Figure 1 is ordered based on the availability of training samples for every class. The shape-sharing CECNN consistently outperforms the baseline Z2CNN, where the average performance of Z2CNN is 45.0% and the average performance of CECNN is 49.1%.

\* *We made the data generation deterministic by setting a seed, and recreating our experiment would return the same data distribution.*


#### Color Selectivity

Color selectivity is defined as: "The property of a neuron that activates highly when a particular color appears in the input image and, in contrast, shows low activation when this color is not present."

In Figure 2, the accuracy improvement of color equivariance up to later stages in the network is displayed for the aforementioned datasets. The baseline is the ResNet-18 model trained for one hue rotation only.

<div align="center">
  <img src="blogpost_imgs/color_selectivity.png" alt="Color selectivity results" width="600">

  *Figure 2: Influence of color equivariance embedded up to late stages in the network on datasets with high and low color selectivity.*
</div>

Similar to the original paper's results, the color-selective dataset benefits from color equivariance up to later stages in the network, in contrast to the less color-selective dataset. This effect is more pronounced in the color-selective dataset, supporting the hypothesis that color equivariance is most beneficial for color-selective tasks.

### Color Equivariance in Image Classification and the Impact of Number of Rotations

We will now explore the reproduction of the main results along with a small insight into the number of rotations. These results are all limited to the Flowers102 dataset since it has the largest color diversity and variability.

#### Image Classification

To evaluate the image classification performance, we trained a baseline ResNet-18 model comprising approximately 11.4M parameters, alongside the novel color equivariant ResNet-18 (CE-ResNet-18) trained with color equivariance of 3 discrete hue rotations.

<div align="center">
  <img src="blogpost_imgs/Test-time_Hue_Shifts.png" alt="Classification Test-time Hue Shifts" width="600">

  *Figure 3: Image classification performance (test accuracy) on flowers classification dataset under a test-time hue shift*
</div>

In Figure 3, both the baseline ResNet-18 and the CE-ResNet-18 demonstrate good performance when no hue shift is applied (test-time hue shift of 0 degrees). The average accuracy across all test-time shifts without jitter training is 73.5% for the baseline and 77.8% for the CE-ResNet-18.

When trained with jitter, both models exhibit robustness against distributional shifts, in line with the original author's findings, with the CE-ResNet-18 showing slightly better performance. This advantage comes at a computational cost.

Comparing training and testing times, the baseline model completes its training approximately 50% faster than the CEConv model. Testing time took around 2.3 times as long for the novel CEConv model. Training took approximately 1.5 hours for the baseline and 3 hours for the CEConv model.

#### Number of Rotations

The main implementation of color equivariance consists of adding three rotations of 120 degrees, whereas the baseline model (without equivariance) can be expressed as having one rotation. In Figure 4, we explore the effect of varying the number of rotations.

<div align="center">
  <img src="blogpost_imgs/rotations.png" alt="Hue rotations" width="600">

  *Figure 4: Effect of the number of rotations on the accuracy of the CE-ResNet-18.*
</div>

While the smooth plot lines of the original paper indicate that they evaluated their models across more discrete hue shifts, the trends are similar to the original paper's findings. The number of parameters grows with the number of rotations, leading to a trade-off between model capacity and equivariance.

| Number of Rotations | Number of Parameters | Max. Accuracy |
|---|---|---|
| 1 | 11.2 M | 70.3% |
| 5 | 11.6 M | 72.4% |
| 10 | 11.8 M | 74.6% |

*Table 1: Parameter and maximum accuracy increase based on number of rotations.*

## Further Research: Extending the Notion of Color Equivariance 

The reproduced results showcase that the notion of equivariance can be extended to photometric transformations by incorporating parameter sharing over hue shifts. However, as opposed to what the title of the original paper suggests, color equivariance is not limited to hue shifts alone.

Additionally, one noticeable flaw in the work of [[5]](#main) is the fact that they model hue shifts with a 3D rotation in the RGB space along the diagonal vector [1,1,1]. This can cause pixels with very high or very low values to clip at the boundaries of the RGB cube.
For an overview of the color spaces and their limitations, we refer to section [Color Spaces](#a-color-spaces) in the Appendix.

### HSV Equivariance

In our implementation of the HSV space, **hue** is modeled as an angular value between 0 and $2\pi$ and can be changed by adding or subtracting such an angle modulo $2\pi$. Therefore, we represent hue shifts as cyclic additions:

$$\begin{align*}
H_n(k) = \begin{bmatrix} \tfrac{2k\pi}{n} \\ 0 \\ 0 \end{bmatrix}
\tag{10}
\end{align*}$$

In which $n$ is the discrete number of rotations and $k$ indicates the $k$-th rotation out of $n$. The group action is an addition on an HSV pixel value in $\mathbb{R}^3$ modulo $2\pi$:

$$\begin{align*}
[H_n(k)f](x) = \begin{bmatrix} \left(f(x)_h + \tfrac{2k\pi}{n}\right) \bmod 2\pi \\ f(x)_s \\ f(x)_v \end{bmatrix}
\tag{11}
\end{align*}$$

with $f(x)_{h,s,v}$ indicating the respective hue, saturation, or value at pixel value $x$ in input image $f$. [[5]](#main) applies this transformation on the kernel made possible by the use of the inverse transformation. However, this creates an inconsistency:

$$\begin{align*}
[H_n(k)f](x) \cdot \psi(y) \neq f(x) \cdot [H_n(-k)\psi](y)
\tag{12}
\end{align*}$$

To see this difference some models are trained in which the kernels are naively shifted as if they were an image and compared to models in which the shift is applied to the images. **Shifting the images works, but shifting the kernels does not.**

We can now define the group $G = \mathbb{Z}^2 \times H_n$ as the product of the 2D integers translation group and the HSV hue shift group. With $\bmod$ as the modulo operator and $\mathcal{L}_{t,m}$ as the group action on translations and hue shifts:

$$\begin{align*}
[\mathcal{L}_{t,m}f](x) = [H_n(m)f](x-t) =
\begin{bmatrix}
\left(f(x-t)_h + \tfrac{2\pi}{n}\,m\right) \bmod 2\pi \\
f(x-t)_s \\
f(x-t)_v
\end{bmatrix}
\tag{13}
\end{align*}$$

We can then define the lifting layer outputting the $i$-th output channel as:

$$\begin{align*}
[\psi^i \star f](x, k) = \sum_{y \in \mathbb{Z}^2} \psi^i(y) \cdot [H_n(k)f](y-x)
\tag{14}
\end{align*}$$

Here $f$ is the input image and $\psi^i$ a set of corresponding filters. Equivariance can be shown as:

$$\begin{align*}
[\psi^i \star [\mathcal{L}_{t,m} f]](x, k)
  &= \sum_{y \in \mathbb{Z}^2} \psi^i(y) \cdot [H_n(k-m)f](y-(x-t)) \\
  &= [\psi^i \star f](x-t,\, k-m) \\
  &= [\mathcal{L}'_{t,m}[\psi^i \star f]](x, k)
\tag{15}
\end{align*}$$

Since the input HSV image is now lifted to the group space, all subsequent features and filters are functions that need to be indexed using both a pixel location and a discrete rotation. The group convolution for hidden layers is:

$$\begin{align*}
[f \star \psi^i](x, k) = \sum_{y \in \mathbb{Z}^2}\sum_{r=1}^{n} f(y,r) \cdot \psi^i\!\left(y-x,\; (r-k) \bmod n\right)
\tag{16}
\end{align*}$$

**Saturation** is represented as a number between 0 and 1, requiring a group that contains $n$ elements equally spaced between -1 and 1 to model both an increase and decrease in saturation. This makes saturation shifts acyclic (as opposed to hue shifts which are cyclic). The saturation group is:

$$\begin{align*}
H_n(k) = \begin{bmatrix} 0 \\ -1 + k\,\tfrac{2}{n-1} \\ 0 \end{bmatrix}
\tag{17}
\end{align*}$$

Because saturation is only defined between 0 and 1 and is acyclic, we clip the value after the group action:

$$\begin{align*}
[H_n(k)f](x) =
\begin{bmatrix}
f(x)_h \\
\text{clip}\!\left(0,\; f(x)_s + \left(-1 + k\,\tfrac{2}{n-1}\right),\; 1\right) \\
f(x)_v
\end{bmatrix}
\tag{18}
\end{align*}$$

This clipping due to the acyclic nature of saturation might break equivariance, which will be tested with several experiments: applying the group action on the kernel and the image (see saturation equivariance results below).

**Value** equivariance can be modeled in the same way as described for saturation where the group action is now acting upon the value channel:

$$\begin{align*}
[H_n(k)f](x) =
\begin{bmatrix}
f(x)_h \\
f(x)_s \\
\text{clip}\!\left(0,\; f(x)_v + \left(-1 + k\,\tfrac{2}{n-1}\right),\; 1\right)
\end{bmatrix}
\tag{19}
\end{align*}$$

Due to our earlier experiments involving the application of the group element on the kernel or the image, we decided to only model the value shift on the input images.


**Combining Multiple Shifts -** Because of the separated channels when utilizing the HSV color space, we can describe the group product between multiple channel shifts as the direct product of the individual groups:

$$\begin{align*}
G = \mathbb{Z}_2 \times C_n \times \mathbb{R} \times \mathbb{R}
\tag{20}
\end{align*}$$

The group action for the corresponding $h'$, $s'$, and $v'$ discrete hue, saturation, and value shifts respectively, is then defined as:

$$\mathcal{L}_{(t,\,h',s',v')} = [H_n(h',s',v')f](x-t) =
\begin{bmatrix}
\left(f(x-t)_h + \tfrac{2\pi}{n}\,h'\right) \bmod 2\pi \\
\text{clip}\!\left(0,\; f(x-t)_s + \left(-1 + s'\tfrac{2}{n-1}\right),\; 1\right) \\
\text{clip}\!\left(0,\; f(x-t)_v + \left(-1 + v'\tfrac{2}{n-1}\right),\; 1\right)
\end{bmatrix}
\tag{21}$$


### LAB Equivariance

Hue equivariance in the LAB color space can be modeled as a 2D rotation on the *a* and *b* channels which represent the color of a LAB pixel. However, due to the differences that arise when converting between color spaces, the exact transformation differs from RGB and HSV.

<div align="center">
  <img src="blogpost_imgs/hue_shift_comparison.png" alt="Hue shift in different image spaces" width="600px">

  *Figure 5: An example image (original far left) hue space shifted multiple times in HSV (angular addition), RGB (3D rotation), and LAB (2D rotation) space, thereafter converted to RGB space for visualization.*
</div>

Figure 5 shows that a hue shift in RGB and HSV space results in the same image. However, performing the same shift in LAB space and converting it back to RGB space afterward, results in a slightly different result due to the non-linear nature of color space conversions.

For the LAB space, only a hue shift equivariant model is implemented. For this, the theory in Section [Color Equivariance](#color-equivariance) is applicable with the only exception being the reparameterization of the rotation matrix to act on the *a* and *b* channels:

$$\begin{align*}
H_n =
\begin{bmatrix}
1 & 0 & 0 \\
0 & \cos\!\left(\tfrac{2k\pi}{n}\right) & -\sin\!\left(\tfrac{2k\pi}{n}\right) \\
0 & \sin\!\left(\tfrac{2k\pi}{n}\right) & \cos\!\left(\tfrac{2k\pi}{n}\right)
\end{bmatrix}
\tag{22}
\end{align*}$$

In which $n$ represents the number of discrete rotations in the group and $k$ indexes the rotation to be applied. The group operation is now a matrix multiplication on the $\mathbb{R}^3$ space of LAB pixels.

### Results of Additional Experiments

The experiments of the various color spaces are conducted on the Flowers102 dataset, similar to section [Image Classification](#image-classification). Furthermore, all experiments are conducted on the same hardware to ensure fair comparison.

#### HSV

This section outlines the results of the equivariance in HSV space. This includes hue shift equivariance applied on the kernel and the image, saturation equivariance on the kernel and image, value equivariance, and combinations thereof.

##### Hue Shift Equivariance

**Shifting the Kernel -** For this experiment, where we naively shift the hue of the input layer filters, we replaced the standard convolutional layers of the ResNet-18 network with our group convolutional layers.
We separate two cases where we train the potential equivariant network (CE-ResNet-18) with hue jitter, randomly applying a hue shift with a uniformly chosen hue factor between -0.5 and 0.5, and without.

<div align="center">
  <img src="blogpost_imgs/HSV_hue_shift_kernel.png" alt="Results of HSV space hue equivariance, when lifting operation is performed by naively hue shifting the kernel" width="600px">

  *Figure 6: Illustrates the test accuracy scores of a variety of models evaluated with 37 test-time hue shifts spanning the full range of -180° to 180° hue rotations. The CE-ResNet-18 models are trained with and without hue jitter.*
</div>

As expected, naively shifting the kernel does not work. In Figure 6, both the CE-ResNet-18 and the baseline model show a peak at the 0° hue shift angle, with performance deteriorating as the hue shift increases. This indicates that neither approach correctly implements hue equivariance through kernel shifting.

**Shifting the Input Image -** Instead of naively hue-shifting the kernel we now perform the lifting convolution by shifting the input image effectively creating a hue-shifted image stack. Thus, we create multiple copies of the input at different hue shifts.

<div align="center">
  <img src="blogpost_imgs/HSV_hue_shift_img.png" alt="Results of HSV space hue equivariance, when lifting operation is performed by hue shifting the input image" width="600px">

  *Figure 7: Illustrates the test accuracy scores of a variety of models evaluated with 37 test-time hue shifts spanning the full range of -180° to 180° hue rotations. The CE-ResNet-18 models are trained with and without hue jitter.*
</div>

Figure 7 shows clear peaks at the 0°, 120°, and 240° (-120°) hue rotation angles for the CE-ResNet-18, effectively exploiting its hue shift equivariance. However, this equivariance is limited to the discrete rotations used during training (multiples of 120 degrees in this case).
For our experiment, we have opted to decrease the network width such that both models have approximately 11.2M parameters, resulting in a less expressive CE-ResNet-18 when compared to the baseline to maintain fairness in comparison.

##### Saturation Equivariance

**Shifting the Kernel -** This experiment largely follows the setup from the hue equivariant network in HSV space. However, five saturation shifts were applied on the kernel and 50 saturation shifts were applied during testing to comprehensively evaluate the model's robustness.

<div align="center">
  <img src="blogpost_imgs/Sat_HSV_Fig9_satshiftKernel.png" alt="Results of HSV space saturation equivariance, when lifting operation is performed by naively saturation shifting the kernel" width="600px">

  *Figure 8: Accuracy over test-time saturation shift for saturation equivariant networks trained using input images in HSV color space format. ResNet-18 indicates a baseline model, CE indicates Color Equivariant model.*
</div>

In Figure 8, we again find that shifting the kernel does not lead to equivariant performance as no clear peaks can be observed. However, the equivariant model outperforms the baseline when no shift occurs, suggesting some benefit even without true equivariance.

**Shifting the Input Image -** In this next approach, the input signal was transformed instead, akin to the hue equivariant version. However, the aforementioned settings for saturation shifts are applied similarly to the hue shifts.

<div align="center">
  <img src="blogpost_imgs/Sat_HSV_Fig9_satshiftImage.png" alt="Results of HSV space saturation equivariance, when lifting operation is performed by saturation shifting the input image" width="600px">

  *Figure 9: Accuracy over test-time saturation shifts for saturation equivariant networks trained using input images in HSV color space format. ResNet-18 indicates a baseline model, CE indicates Color Equivariant model.*
</div>

As opposed to hue equivariance, shifting the image does not lead to equivariant performance for saturation (Figure 9). We suspect that this occurs due to the clipping and the non-cyclic nature of saturation, which breaks the equivariance property.

Furthermore, both experiments were conducted on the Camelyon17 dataset as well. Similar trends were observed, which are discussed in Appendix [Saturation Equivariance Camelyon17](#e-saturation-equivariance-camelyon17).

##### Value Equivariance

For value equivariance, we only tested shifting the input images. Initially, we trained with five value shifts in a range from -1 to 1. However, this results in totally black images with a complete loss of information at the extremes.

<div align="center">
  <img src="blogpost_imgs/value_equivariance.png" alt="HSV space value equivariance" width="600px">

  *Figure 10: Accuracy over test-time value shifts for value equivariant networks trained using input images in HSV color space format. ResNet-18 indicates a baseline model, CE indicates Color (value) Equivariant model.*
</div>

While being trained with five different shifts the model is not able to show equivariance, similar to the saturation results, and follows the performance of the baseline ResNet-18. Training with jitter across a smaller range of -0.5 to 0.5 improves results but still does not achieve clear equivariant behavior.

##### Combining Hue and Saturation Equivariance

**Shifting the Kernel -** A hue and saturation equivariant kernel was created for any combination of three hue and saturation shifts. Furthermore, the baseline model was again ResNet-18.

<p align="middle">
  <img src="blogpost_imgs/HueSat_HSV_shiftKernelBase_noNorm.jpg" alt="ResNet-18 model tested on Hue and Saturation equivariance in HSV space" width="49%">
  <img src="blogpost_imgs/HueSat_HSV_shiftkernel_noNorm.jpg" alt="Hue and Saturation equivariance in HSV space" width="49%">
</p>

*Figure 11: On the left, the baseline model's test accuracy is calculated over varying hue and saturation shifts. On the right, the hue and saturation equivariant network's test accuracy is displayed over the same range of transformations.*

On the left in Figure 11, it is shown that the baseline achieves the same test accuracies as when the shifts were applied independently of each other. This is also true for hue shifts for the hue and saturation equivariant model on the right, showing clear peaks at hue shift multiples of 120°.

**Shifting the Input Image -** For this experiment, the same combinations of hue and saturation shifts were utilized as in the previous experiment. However, to lift the input image to the group we transform both hue and saturation simultaneously.

<p align="middle">
  <img src="blogpost_imgs/HueSat_HSV_shiftImgBase_noNorm.jpg" alt="ResNet-18 model tested on Hue and Saturation equivariance in HSV space with transformations applied to the input image" width="49%">
  <img src="blogpost_imgs/HueSat_HSV_shiftimg_noNorm.jpg" alt="Hue and Saturation equivariance in HSV space with transformations applied to the input image" width="49%">
</p>

*Figure 12: On the left, the baseline model's test accuracy is calculated over varying hue and saturation shifts. On the right, the hue and saturation equivariant network's test accuracy is displayed over the same range of transformations.*

The network that combines hue and saturation equivariance on the right of Figure 12 does not gain additional performance by this combination when compared to the results of the individually equivariant models. The hue equivariance is preserved (peaks at 120° intervals), but the saturation equivariance does not manifest.

Ultimately, the only improvement from these experiments was for saturation equivariance when both hue and saturation equivariances were applied to the kernel. Although this result seems promising, we cannot reliably reproduce this finding, suggesting it may be a statistical artifact.

#### LAB

To test hue equivariance implemented in LAB space the convolution layers of a ResNet-18 network were replaced by their equivariant counterpart. The equivariant layers are implemented using three discrete hue rotations (120° apart).

During test time, different sets of hue-shifted images are evaluated on accuracy. This hue shift is either done in RGB space after which the RGB images are converted to LAB format, or directly in LAB space before converting back to RGB for visualization and evaluation.

<div align="center">
  <img src="blogpost_imgs/lab_equivariance.png" alt="LAB space hue equivariance" width="600px">

  *Figure 13: Accuracy over test-time hue shifts for hue equivariant networks trained using input images in LAB color space format. ResNet-18 indicates a baseline model, CE indicates Color (hue) Equivariant model.*
</div>

Figure 13 displays that the hue equivariant network (CE) tested with hue space shifts in RGB/HSV space shows small bumps around ±120°, demonstrating a slight improvement over the ResNet-18 baseline. However, the effect is less pronounced than in the HSV case.


### Comparison of different color spaces

This blog post explored three different color spaces to represent hue (and saturation and value) equivariance. In Figure 14 we display the color equivariant models trained in the different color spaces, evaluated both with and without jitter training.

<div align="center">
  <img src="blogpost_imgs/comparison.png" alt="color equivariant models trained in different color spaces" width="600">

  *Figure 14: Color equivariant models trained in different color spaces and tested in the RGB color space, with and without jitter.*
</div>

The results illustrate that the model trained in the RGB color space has the best overall performance when trained without jitter. The LAB model does not display equivariant properties on the hue-shifts as clearly as the HSV model.

When including jitter, the LAB space model outperforms both the RGB and HSV space models. Training the LAB color equivariant model with jitter results in an increased average accuracy of six percentage points compared to training without jitter. This suggests that LAB color equivariance combined with data augmentation is particularly effective.

## Concluding Remarks

In conclusion, the network proposed in [[5]](#main) aimed to leverage color equivariance to create more robust networks that still manage to exploit color information from images. This was implemented through group equivariant convolutions in RGB space.

Additionally, we investigated the limitations of the approach taken by the original authors. Firstly, the limited notion of modeling color equivariance as hue equivariance. Secondly, the problem of pixel values clipping at RGB boundaries.
We found that a model trained for hue equivariance in LAB space with hue jitter managed to outperform all other models that encode this type of equivariance. Furthermore, saturation and value equivariance showed limited effectiveness, likely due to the non-cyclic and clipping operations required.

In future research, the use of steerable convolutions could be explored. Steerable convolutions could be used to encode equivariance to the continuous hue spectrum, therefore achieving equivariance without discretizing the hue rotations.

## Authors' Contributions

- **Silvia Abbring:** Implementation of saturation equivariance and combining hue and saturation shifts during testing, wrote concluding remarks, results of combining hue and saturation shifts, and appendices B and C.
- **Hannah van den Bos:** Reproduction of color selectivity, rotation, and jitter ablation with implementation of plots and supplementary function evaluate, wrote introduction, recap on group equivariant convolutions, and authors' contributions section.
- **Rens den Braber:** Implementation/Description of LAB space and Value equivariance, and HSV equivariance formulas.
- **Arco van Breda:** Reproduction of color imbalance and image classification, implementation of reproducibility plots and supplementary functionalities (load, save, evaluate) in the original code, and appendix D.
- **Dante Zegveld:** Implementation of hue shift equivariance with lifting performed on the kernel but also on the image. Implementation of saturation shift equivariance when lifting the image and comparison plots.

## References

<a id="group_convs">[1]</a> Cohen, T. & Welling, M. (2016). Group Equivariant Convolutional Networks. *Proceedings of The 33rd International Conference on Machine Learning*, in *Proceedings of Machine Learning Research*.

<a id="color_net">[2]</a> Gowda, S. N., & Yuan, C. (2019). ColorNet: Investigating the importance of color spaces for image classification. In *Computer Vision – ACCV 2018*, Part IV (pp. 581-596). Springer.

<a id="color_selectivity">[3]</a> Ivet Rafegas and Maria Vanrell. Color encoding in biologically-inspired convolutional neural networks. *Vision Research*, 151:7–17, 2018.

<a id="human_vision">[4]</a> Ivet Rafegas, Maria Vanrell. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 2697-2705.

<a id="main">[5]</a> Lengyel, A., Strafforello, O., Bruintjes, R. J., Gielisse, A., & van Gemert, J. (2024). Color Equivariant Convolutional Networks. *Advances in Neural Information Processing Systems*, 36.

<a id="color_segmentation">[6]</a> Raninen, J. (2022). The Effect of Colour Space in Deep Multitask Learning Neural Networks for Road Segmentation (Master's thesis, Itä-Suomen yliopisto).

<a id="color_invariance">[7]</a> R. Rama Varior, G. Wang, J. Lu and T. Liu, "Learning Invariant Color Features for Person Reidentification," in *IEEE Transactions on Image Processing*, vol. 25, no. 7, pp. 3395-3410, July 2016, doi: 10.1109/TIP.2016.2564142.

<a id="bird">[8]</a> Simen Hagen, Quoc C. Vuong, Lisa S. Scott, Tim Curran, James W. Tanaka. The role of color in expert object recognition. *Journal of Vision* 2014;14(9):9. https://doi.org/10.1167/14.9.9.

<a id="lifting">[9]</a> Worrall, D., & Welling, M. (2019). Deep scale-spaces: Equivariance over scale. *Advances in Neural Information Processing Systems*, 32.

<a id="DCNN">[10]</a> W. Rawat and Z. Wang, "Deep Convolutional Neural Networks for Image Classification: A Comprehensive Review," in *Neural Computation*, vol. 29, no. 9, pp. 2352-2449, Sept. 2017, doi: 10.1162/neco_a_00990.

## Appendices

### A. Color Spaces

While most CNNs are trained using RGB images, work by [[2]](#color_net) and [[6]](#color_segmentation) shows that different color spaces can be utilized to achieve similar performance for the task of image classification. Each color space has different properties and trade-offs.

**RGB** - is the most frequently used color space in image datasets. However, it is limited by the above mentioned clipping effects near the boundaries of the RGB cube. Furthermore, due to the entanglement of color information across channels, it is not ideal for separating luminance and chrominance information.

<p align="center">
  <img alt="A visualization of the RGB color space" src="blogpost_imgs/RGB_CUBE_ROT_AXIS.png" width="250px">
  &nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="A visualization of the HSV color space" src="blogpost_imgs/HSV_COLOR_SPACE.png" width="350px">
</p>

*Figure A.1: The left figure displays the RGB color space as a cube where the rotational axis (Rot-Axis [1,1,1]) along which hue rotations are modeled in this space is also illustrated. The right figure shows the HSV color space visualized as a cylinder.*

**HSV** - is an ideal color space for our purpose of extending the achieved hue equivariant CNN with saturation equivariance. With a separate channel encoding the hue of each pixel, we can make a direct correspondence between rotations and hue shifts.
However, there are some potential issues with this color space. Firstly, the hue channel, in our implementation, is encoded as an angle ranging from $0$ to $2\pi$. Although these values encode the same color at 0 and $2\pi$, this discontinuity can cause issues in some applications.

**LAB** - is a color space defined by the International Commission on Illumination (CIE) in 1976. Research by [[2]](#color_net) and [[6]](#color_segmentation) shows that images converted to LAB color space can lead to improved robustness in certain computer vision tasks. The LAB color space is designed to approximate human vision.

<div align="center">
  <img src="blogpost_imgs/lab-color-space.png" alt="Visualization of the LAB color space" width="600px">

  *Figure A.2: left: LAB color space visualized as a 2d color grid, right: sRGB color gamut shown in LAB space. ([source](https://www.xrite.com/blog/lab-color-space), [source](https://blogs.mathworks.com/steve/2010/07/08/lab-color-space/), [source](https://en.wikipedia.org/wiki/Gamut))*
</div>

### B. Ablation Study Saturation Equivariance

Further investigation was conducted on the impact of the number of shifts and the degree of jitter to obtain saturation equivariance. The results will be discussed here and are employed in the experimental setup.

**Effects of Saturation Shifts -** For the number of shifts, different models were trained with respectively 3, 5, and 10 saturation shifts ranging from -1 to 1 and including 0, so no shift. The baseline model was a standard ResNet-18 without saturation equivariance.

<div align="center">
  <img src="blogpost_imgs/Sat_HSV_ShiftsKernel.png" alt="Saturation shift ablation" width="70%">

  *Figure B.1: Test accuracy of a saturation equivariant model trained on a varying number of saturation shifts ranging from -1 to 1 while including 0. The baseline is indicated with None. No jitter was applied during training.*
</div>

Figure B.1 showcases that the number of shifts does not make a significant impact, since all saturation equivariant networks obtain approximately equal performance. Therefore, we opted to go for the most efficient choice of 5 saturation shifts in subsequent experiments.

**Effects of Saturation Jitter -** Saturation jitter was implemented by using Pytorch's function which is called directly when the data is loaded in. However, a disadvantage is that this implementation applies jitter to the entire image uniformly.

<div align="center">
  <img src="blogpost_imgs/Sat_HSV_satshiftkernel_jitter.png" alt="Saturation jitter ablation" width="70%">

  *Figure B.2: Test accuracy of a saturation equivariant model. The model was trained on 5 saturation shifts, namely -1, -0.5, 0, 0.5 and 1. The baseline model (None) is a CE-ResNet-18 model trained without any saturation jitter.*
</div>

In the above figure, all degrees of saturation jitter enhance robustness to test-time saturation distribution shifts compared to the baseline with none. The upper bound of 2 ensures an increased test-time saturation range beyond the training range, better testing generalization.

### C. Reproduction of Jitter Ablation Study

Figure 3 seems to suggest that solely adding color-jitter augmentation to the ResNet18 model is sufficient for high accuracy. However, implementing it along with the CE-ResNet18 model seems complementary, as both benefit from jitter.

In the original paper, color-jitter augmentation is limited to randomly changing the hue of an image, leaving the brightness, contrast, and saturation unchanged. Setting the (hue) jitter value to 0.5 means that the hue is changed uniformly at random with a probability of 0.5 by up to ±0.5 of the full hue range.

Figure C.1 displays a more nuanced view of the jitter, showing the ResNet18 model with jitter values 0.2 and 0.4, and the CE-ResNet18 model with jitter values 0.1 and 0.2. Moreover, the baseline CE-ResNet-18 model trained without jitter is also shown for comparison.

<div align="center">
  <img src="blogpost_imgs/jitter.png" alt="Jitter ablation" width="600px">

  *Figure C.1: Test accuracy over the hue-shift for color-equivariant and ResNet-18 with various degrees of color-jitter augmentation.*
</div>

The figure shows that in order to create stable accuracy for the original model, jitter values of 0.2 and 0.4 are insufficient. Instead, the jitter augmentation should at least account for fluctuations across the discrete hue rotations used in the equivariant model to achieve consistent performance.

### D. Training time study

During the reproduction of "Color Imbalance" we observed a significant discrepancy in the training times required for the two models. To verify that the CECNN excels in retaining and sharing shape information without introducing excessive computational overhead, we trained a standard CNN with increased model capacity.

<div align="center">
  <img src="blogpost_imgs/Longtailed_appendix.png" alt="Training time study" width="600px">

  *Figure D.1: Classification performance of a standard CNN (Z2CNN) and the color equivariant convolutions CNN (CECNN) on a long-tailed, unequally distributed dataset. Additionally, a Z2CNN with increased width (2x) is shown for comparison.*
</div>

The figure clearly demonstrates even when training a standard CNN model with significantly more parameters and comparable training time, the CECNN consistently outperforms both models. The CNN model with 2x width has approximately 50,000 parameters compared to CECNN's roughly 25,990 parameters, yet CECNN achieves better generalization on this color-imbalanced task.

### E. Saturation Equivariance Camelyon17

In order to further explore the cause for the trends as observed in Figures 8 and 9, we ran the same experiments on the Camelyon17 dataset. Namely, adjusting the saturation of flowers could change the appearance significantly, whereas medical images might be less affected by saturation changes.

**Shifting the Kernel -** This experiment followed the same setup as the saturation equivariant one with the Flowers102 dataset. The results are displayed in the figure below.

<div align="center">
  <img src="blogpost_imgs/Sat_HSV_Fig9_satshiftKernel_Camelyon17.jpg" alt="Saturation equivariance on Camelyon17 dataset with transformations applied on kernel" width="70%">

  *Figure E.1: Test accuracy over test-time saturation shift for saturation equivariant networks trained using input images from Camelyon17 dataset in HSV color space format. ResNet-18 indicates a baseline model, CE indicates Color Equivariant model.*
</div>

The figure above displays a clear peak around no saturation shifts for all settings, indicating a lack of equivariance. Nonetheless, CE-ResNet-18 outperforms ResNet-18 on average. Furthermore, the model with jitter shows improved robustness across saturation shifts compared to the model without jitter.

**Shifting the Input Image -** An equivalent setup to the saturation equivariant network trained on the Flowers102 dataset with the transformations applied to the input image was chosen in order to compare results.

<div align="center">
  <img src="blogpost_imgs/Sat_HSV_Fig9_satshiftImage_Camelyon17.png" alt="Saturation equivariance on Camelyon17 dataset with transformations applied on image" width="70%">

  *Figure E.2: Test accuracy over test-time saturation shift for saturation equivariant networks trained using input images from Camelyon17 dataset in HSV color space format. ResNet-18 indicates a baseline model, CE indicates Color Equivariant model.*
</div>

Figure E.2 is similar to E.1. A difference concerns the peaks: they have become narrower for all models, indicating a lack of equivariance. However, CE-ResNet-18 without jitter obtained a higher test accuracy at zero saturation shift (around 83%) compared to the Flowers102 dataset.

Generally, the trends are similar to the ones on the Flowers102 dataset. This could indicate that the saturation equivariance gets broken in the network. We hypothesize this is due to the clipping operations and the non-cyclic nature of saturation, which fundamentally breaks the group equivariance property in deep networks.
