use image::{ImageBuffer};
use image::{Rgb, Rgba, Luma, Bgr, Bgra, LumaA, Pixel, ColorType};

use ndarray::{Array2, Array1, arr2, arr1, ArrayView, ArrayBase, ViewRepr, Axis};
use ndarray_linalg::eig;
use rand_distr::{Distribution, Normal, NormalError};
use rand::thread_rng;
use rand::{RngCore, rngs::ThreadRng};
use ndarray::*;
use ndarray_linalg::*;



#[derive(Debug)]
struct Error(String);


macro_rules! impl_error_from {
    ($type:path) => {
        impl From<$type> for Error {
            fn from(error: $type) -> Self
            {
                Self(format!("{:?}", error))
            }
        }
    };
    (&$type:path) => {
        impl From<&$type> for Error {
            fn from(error: &$type) -> Self
            {
                Self(format!("{:?}", error))
            }
        }
    };
}

impl_error_from!(image::ImageError);
impl_error_from!(ndarray::ShapeError);
impl_error_from!(ndarray_linalg::error::LinalgError);
impl_error_from!(rand_distr::NormalError);


#[derive(Debug, Clone)]
struct Image(Array2<f32>);
impl Image
{
    fn from(img: image::DynamicImage) -> Self
    {
        let img = img.to_luma();
        let w = img.width();
        let h = img.height();

        let mut mat = Array2::<f32>::zeros((h as usize, w as usize));            

        for i in 0..w
        {
            for j in 0..h
            {
                let image::Luma([v]) = img.get_pixel(i, j);
                mat[[j as usize, i as usize]] = ((*v) as f32)/255.0;
            }
        }
        Self(mat)
    }

    fn from_mat(mat: Array2<f32>) -> Self
    {
        Self(mat)
    }
    fn new(name: &str) -> Result<Self, Error>
    {
        Ok(Self::from(image::open(name)?))
    }

    fn save(&self, name: &str) -> Result<(), Error>
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

    fn h(&self) -> usize
    {
        self.0.nrows()
    }

    fn w(&self) -> usize
    {
        self.0.ncols()
    }

    fn len(&self) -> usize
    {
        self.w()*self.h()
    }

    /**
    Used to safely read the pixels of the image.
    If the coordinates are out of bound, it mirrors the image.
    If the out of bound is too great (more than the width or height of the image), this method panics.
    This is not a problem since the patches will not have a radius equal or supperior to the dimensions of the image.
     */
    fn get(&self, i: isize, j: isize) -> f32
    {
        let (i, j) = ( i.max(-i).min(2*self.w() as isize -1 - i) as usize,
                       j.max(-j).min(2*self.h() as isize -1 - j) as usize);
        self.0[[j, i]]
    }

    fn coord_to_index(&self, i: usize, j: usize) -> usize
    {
        self.w()*j+i
    }

    fn index_to_coord(&self, i: usize) -> (usize, usize)
    {
        (i % self.w(), i / self.w())
    }        
    
    fn patches(&self, radius: usize) -> Array2<f32>
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
    fn snr(&self, reference: &Image) -> f32
    {
        let dist = (self.0.clone() - reference.0.clone()).fold(0f32, |sum, x| x*x).sqrt();
        let norm = self.0.fold(0f32, |sum, x| x*x).sqrt();

        return 20.0*(norm/dist).log10();
        
    }

    fn gaussian_noise(&self, sigma: f32) -> Result<Self, Error>
    {
        let mut rng = rand::thread_rng();
        let gaussian = Normal::new(0f32, sigma)?;
        Ok(Self(self.0.map(|x| x + gaussian.sample(&mut rng))))
    }
    
}


fn range2d(istart: isize, jstart: isize, iend: isize, jend: isize) -> Vec<(isize, isize)>
{
    (istart..iend).map(move |i| (jstart..jend).map(move |j| (i, j)))
        .flatten()
        
        .collect::<Vec<_>>()
}




fn main() -> Result<(), Error>
{
    let dist_radius = 8;
    let radius = 1;

    let tau = 0.05;




    let img_ref = Image::new("images/ricardo_power.png")?;

    let img = img_ref.gaussian_noise(0.05)?;

    img.save("noisy.png")?;
    
    println!("Image shape: {:?}", img.0.shape());
    println!("w {}  h {}", img.w(), img.h());
    let diam = 2*radius+1; 
   
    let patches = img.patches(radius);
    let mat_patches_shape = (img.len(), diam*diam);
    let patch_mean = patches.mean_axis(Axis(0)).unwrap();

    println!("shape patches {:?}", patches.shape());
    let patches_centered = patches - patch_mean.broadcast(mat_patches_shape).unwrap();
    let c = patches_centered.t().dot(&patches_centered);

    println!("shape c {:?}", c.shape());

    let (eig_values_complex, eig_vecs_complex) = Eig::eig(&c)?;

    println!("EIGVEC SHAPE: {:?}", eig_vecs_complex.shape());
    
    let eig_values_real = eig_values_complex.map(|c| c.re);
    let eig_vecs_real = eig_vecs_complex.map(|c| c.re);

    //    let mut eig_tuples = eig_values_real.iter().zip(eig_vecs_real.iter())
    //      .collect::<Vec<_>>();
    let mut eig_tuples = (0..diam*diam).map(|i| (eig_values_real[i], eig_vecs_real.column(i).to_owned())).collect::<Vec<_>>();
    
    eig_tuples.sort_by(|(val1, vec1), (val2, vec2)| val2.partial_cmp(val1).unwrap());

    let total_val: f32 = eig_values_real.iter().sum();
    let treshold = total_val/50.;
    let mut dim = diam*diam;
    let mut sum_val = total_val;
    for i in 0..(diam*diam)
    {
        if sum_val <= treshold
        {
            dim = i;
            break;
        }
        else
        {
            sum_val -= eig_values_real[i];
        }
    }

    println!("Subspace dim={}", dim);
    let (vals, vecs): (Vec<_>, Vec<_>) = eig_tuples.into_iter().take(dim).unzip();
    println!("Sorted eigen values and vecs");
    for (val, vec) in vals.iter().zip(vecs.iter())
    {
        println!("{:?}     |     {:?}", val, vec);
    }

    let subspace_raw = vecs.iter().map(|vec| vec.iter()).flatten().cloned().collect::<Vec<f32>>();
    let subspace = Array2::from_shape_vec((dim, diam*diam), subspace_raw)?;



    println!("SUBSPACE MATRIX SHAPE: {:?}", subspace.shape());

    let projection = patches_centered.dot(&subspace.t());

    println!("Projection (H) shape: {:?}", projection.shape());

    let projection_reshaped = projection.into_shape((img.h(), img.w(), dim))?;
    
    println!("Projection reshaped into {:?}", projection_reshaped.shape());


    
    let denoized = Image(Array2::from_shape_fn(
        (img.h(), img.w()),
        |(j0, i0)|
        {
            let patch = projection_reshaped.select(Axis(0), &[j0]).select(Axis(1), &[i0]);
            let x = ((0.max((i0 as isize - dist_radius) as usize))..img.w().min((i0 as isize+dist_radius+1) as usize)).collect::<Vec<_>>();
            let y = ((0.max((j0 as isize - dist_radius) as usize))..img.h().min((j0 as isize+dist_radius+1) as usize)).collect::<Vec<_>>();
            let selection = projection_reshaped.select(Axis(0), y.as_slice()).select(Axis(1), x.as_slice())
                - patch.broadcast((y.len(), x.len(), dim)).unwrap();
            let dist = selection.fold_axis(Axis(2), 0f32, |sum, x| sum+x*x)/ ((diam*diam) as f32);
            let prekernel = dist.map(|x| (-x/(2.0*tau*tau)).exp());
            let norm = prekernel.sum();
            let kernel = prekernel/norm;
            let im_selec = img.0.select(Axis(0), y.as_slice()).select(Axis(1), x.as_slice());
            (kernel*im_selec).sum()
        }
   ));

    println!("{:?}", denoized);
    println!("OUTPUT SHAPE {:?}", denoized.0.shape());

    denoized.save("test.png");

    println!("SNR DENOIZED: {}", denoized.snr(&img_ref));
    println!("SNR NOISY   : {}", img.snr(&img_ref));


    
    Ok(())
}
