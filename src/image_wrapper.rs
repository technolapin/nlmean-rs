use image::{ImageBuffer};
use image::{Luma};

use ndarray::{Array1, Array2, Axis};
use rand_distr::{Distribution, Normal};
use ndarray_linalg::*;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use rand::Rng;
use rand::prelude::SliceRandom;
use std::time::Instant;
use std::cmp::Ordering;



use crate::{Float, Error, pca, range2d};

#[derive(Debug, Clone)]
pub struct Image(pub Array2<Float>);
impl Image
{
    /// creates an Image from a image::Image
    pub fn from(img: image::DynamicImage) -> Self
    {
        let img = img.to_luma8();
        let w = img.width();
        let h = img.height();

        let mut mat = Array2::<Float>::zeros((h as usize, w as usize));            

        for i in 0..w
        {
            for j in 0..h
            {
                let image::Luma([v]) = img.get_pixel(i, j);
                mat[[j as usize, i as usize]] = ((*v) as Float)/255.0;
            }
        }
        Self(mat)
    }

    /// creates an image from a matrix
    pub fn from_mat(mat: Array2<Float>) -> Self
    {
        Self(mat)
    }

    /// loads an image
    pub fn new(name: &str) -> Result<Self, Error>
    {
        Ok(Self::from(image::open(name)?))
    }

    /// creates a black image
    pub fn empty(w: usize, h: usize) -> Self
    {
        Self(Array2::default((h, w)))
    }
    
    /// exports the image
    pub fn save(&self, name: &str) -> Result<(), Error>
    {
        let w = self.w();
        let h = self.h();
        let mut img = ImageBuffer::from_pixel(w as u32, h as u32, Luma([0u8]));

        for i in 0..w
        {
            for j in 0..h
            {
                let v = self.0[[j, i]]*255.0;
                img.put_pixel(i as u32, j as u32, Luma([v as u8]));
            }
        }

        img.save(name)?;
        Ok(())
    }

    pub fn h(&self) -> usize
    {
        self.0.nrows()
    }

    pub fn w(&self) -> usize
    {
        self.0.ncols()
    }

    pub fn len(&self) -> usize
    {
        self.w()*self.h()
    }

    /**
    Used to safely read the pixels of the image.
    If the coordinates are out of bound, it mirrors the image.
    If the out of bound is too great (more than the width or height of the image), this method panics.
    This is not a problem since the patches will not have a radius equal or supperior to the dimensions of the image.
     */
    pub fn get(&self, i: isize, j: isize) -> Float
    {
        let (i, j) = ( i.max(-i).min(2*self.w() as isize -1 - i) as usize,
                       j.max(-j).min(2*self.h() as isize -1 - j) as usize);
        self.0[[j, i]]
    }


    /// extracts patches from the image
    pub fn patches(&self, radius: usize) -> Array2<Float>
    {
        let r = radius as isize;

        // the neighborhood determining the shape of the patches
        let ngh = range2d(-r, -r, r+1, r+1);

        let dim_patches_mat = (self.len(), ngh.len());

        let v = range2d(0, 0, self.w() as isize, self.h() as isize)
            .iter()
            .map(|(i, j)| ngh.iter()
                 .map(move |(di, dj)| self.get(i+di, j+dj))
            )
            .flatten()
            .collect::<Vec<_>>();

        // might panic if I made a mistake on the indexes
        Array2::from_shape_vec(dim_patches_mat, v).unwrap()
        
    }

    /**
    Projects the patches of the images on a smaller space choosen using PCA.
    This results on some pseudo-patches that are much smaller than the original spaces with minimal loss of information.
     */
    pub fn projection(&self, radius: usize, tresh: Float) -> Result<(Array2<Float>, Array2<Float>, Array1<Float>, usize), Error>
    {
        let im_len = self.len();
        let pa_len = (radius*2+1).pow(2);
        
        let patches = self.patches(radius);
        let mat_patches_shape = (im_len, pa_len);
        let patch_mean = patches.mean_axis(Axis(0)).unwrap();

        let patches_centered = patches.clone() - patch_mean.broadcast(mat_patches_shape).unwrap();
        
        let (eig_values, eig_vecs) = pca(patches_centered.clone())?;
        
        let total_val: Float = eig_values.iter().sum();
        let treshold = total_val*tresh;
        let mut dim = pa_len;
        let mut sum_val = total_val;
        for i in 0..pa_len
        {
            if sum_val <= treshold
            {
                dim = i;
                break;
            }
            else
            {
                sum_val -= eig_values[i];
            }
        }

        let vecs = eig_vecs.into_iter().take(dim).collect::<Vec<_>>();

        let subspace_raw = vecs.iter().map(|vec| vec.iter()).flatten().cloned().collect::<Vec<Float>>();
        let subspace = Array2::from_shape_vec((dim, pa_len), subspace_raw)?;



        let projection = patches_centered.dot(&subspace.t());

        Ok((projection, patches, patch_mean, dim))
    }
        

    /// Computes the signal-noise ratio, using the reference noiseless image given as parameter
    pub fn snr(&self, reference: &Image) -> Float
    {
        let pow_noise = (self.0.clone() - reference.0.clone()).map(|x| x*x).sum().sqrt();
        let pow_signal = reference.0.map(|x| x*x).sum();

        return 20.0*(pow_signal/pow_noise).log10();
        
    }

    /// Adds a gaussian noise to the image
    pub fn gaussian_noise(&self, sigma: Float) -> Result<Self, Error>
    {
        let mut rng = rand::thread_rng();
        let gaussian = Normal::new(0., sigma)?;
        Ok(Self(self.0.map(|x| x + gaussian.sample(&mut rng))))
    }

    #[allow(dead_code)]
    pub fn crop(&self, x: usize, y: usize, w: usize, h: usize) -> Self
    {
        let xs = (x..(x+w)).collect::<Vec<_>>();
        let ys = (y..(y+h)).collect::<Vec<_>>();
        Self(self.0.select(Axis(0), ys.as_slice()).select(Axis(1), xs.as_slice()))
    }

    /// used to augment the resolution of an image in order
    #[allow(dead_code)]
    pub fn hyper_sample(&self, scale: usize) -> Self
    {
        let mut out = Self::empty(self.w()*scale, self.h()*scale);
        for i in 0..self.w()
        {
            for j in 0..self.h()
            {
                let val = self.get(i as isize, j as isize);
                for di in 0..scale
                {
                    for dj in 0..scale
                    {
                        out.0[[j*scale+dj, i*scale+di]] = val;
                    }
                }
            }
        }
        out
    }

    /**
    concatenate at the right
    panics if sizes are not compatibles
     */
    pub fn concat(&self, other: &Self) -> Self
    {
        // the ndarray::concatenate function will be available in the next version
        // for now I'll just improvise
        assert_eq!(self.h(), other.h());
        let (w, h) = (self.w()+other.w(), self.h());
        let mut new = Self::empty(w, h);

        for i in 0..self.w()
        {
            for j in 0..h
            {
                new.0[[j, i]] = self.0[[j, i]];
            }
        }
        for i in self.w()..w
        {
            for j in 0..h
            {
                new.0[[j, i]] = other.0[[j, i-self.w()]];
            }
        }

        new
    }

    /**
    concatenate at the bottom
    panics if sizes are not compatibles
     */
    pub fn concat_vertical(&self, other: &Self) -> Self
    {
        // the ndarray::concatenate function will be available in the next version
        // for now I'll just improvise
        assert_eq!(self.w(), other.w());
        let (w, h) = (self.w(), self.h()+other.h());
        let mut new = Self::empty(w, h);

        for i in 0..self.w()
        {
            for j in 0..self.h()
            {
                new.0[[j, i]] = self.0[[j, i]];
            }
        }
        for i in 0..self.w()
        {
            for j in self.h()..h
            {
                new.0[[j, i]] = other.0[[j-self.h(), i]];
            }
        }

        new
    }

}
