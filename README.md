## About The Project

Structural similarity [] is a way to compare two images - it's been used as a reliable and fast tracking method []. First, I provide a number of implementation details which speed up the algorithm mathematically (i.e. without touching a line of code) if you are using it for tracking. Second, I provide a numba-optimized Python implementation. At time of writing, it is ()x faster than the version given in scikit-image.


### Quickstart

1. Install necessary packages
   ```sh
   pip install requirements.txt
   ```
2. Clone the repo
   ```sh
   git clone https://github.com/camden-cummings/zf_CV_comp
   ```

## Speed Comparison

## Usage
```

```

## Reasoning
A simplified version of SSIM (assuming $$\alpha = \beta = \gamma = 1$$ and $$C3 = \frac{C2}{2}$$ as in [])  can be implemented as such:

$\mu_x = mean(im_1)$

$\mu_{xx} = mean((im_1)^2)$

$\mu_{xy} = mean(im_1im_2)$

$\mu_y = mean(im_2)$

$\mu_{yy} = mean((im_2)^2)$
\
\
\
$\sigma_x = (cov)(\mu_{xx} - \mu_x^2)$

$\sigma_y = (cov)(\mu_{yy} - \mu_y^2)$

$\sigma_{xy} = (cov)(\mu_{xy} - \mu_y\mu_x)$ 
\
\
\
$L =$ max data range (i.e. 255), $K1 = 0.01$, $K2 = 0.03$

$C1 = (K1 * L)^2 $

$C2 = (K2 * L)^2 $

$$diff = \frac{(2\mu_x\mu_y + C1)(2\sigma_{xy} + C2)}{(\mu_x^2 + \mu_y^2 + C1)(\sigma_x + \sigma_y + C2)}$$ 

When using SSIM for tracking on a video, you can improve speed by assuming one of two things - either you compare against a mode image or compare the previous image taken to the current image taken. Given this implementation, it's clear that $\mu_x$, $\mu_{xx}$, & $\sigma_x$ are all derived from im1 alone, meaning that if im1 is a mode image, they do not need to be recalculated. By the same trick - if you want to compare each previous frame to each current frame, taking im1 as the current image, $\mu_y -> \mu_x$, $\mu_{yy} -> \mu_{xx}$, & $\sigma_y -> \sigma_x$. 

## Acknowledgments

* 



