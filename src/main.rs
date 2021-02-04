use image::{ImageBuffer};
use image::{Luma};

use ndarray::{Array1, Array2, Axis};
use rand_distr::{Distribution, Normal};
use ndarray_linalg::*;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;

type float = f64;

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
struct Image(Array2<float>);
impl Image
{
    fn from(img: image::DynamicImage) -> Self
    {
        let img = img.to_luma8();
        let w = img.width();
        let h = img.height();

        let mut mat = Array2::<float>::zeros((h as usize, w as usize));            

        for i in 0..w
        {
            for j in 0..h
            {
                let image::Luma([v]) = img.get_pixel(i, j);
                mat[[j as usize, i as usize]] = ((*v) as float)/255.0;
            }
        }
        Self(mat)
    }

    fn from_mat(mat: Array2<float>) -> Self
    {
        Self(mat)
    }
    fn new(name: &str) -> Result<Self, Error>
    {
        Ok(Self::from(image::open(name)?))
    }

    fn empty(w: usize, h: usize) -> Self
    {
        Self(Array2::default((h, w)))
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
    fn get(&self, i: isize, j: isize) -> float
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
    
    fn patches(&self, radius: usize) -> Array2<float>
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
    fn snr(&self, reference: &Image) -> float
    {
        let pow_noise = (self.0.clone() - reference.0.clone()).map(|x| x*x).sum().sqrt();
        let pow_signal = reference.0.map(|x| x*x).sum();

        //println!("POW NOISE {}     POW SIGNAL {}", pow_noise, pow_signal).sqrt();
        return 20.0*(pow_signal/pow_noise).log10();
        
    }

    fn gaussian_noise(&self, sigma: float) -> Result<Self, Error>
    {
        let mut rng = rand::thread_rng();
        let gaussian = Normal::new(0., sigma)?;
        Ok(Self(self.0.map(|x| x + gaussian.sample(&mut rng))))
    }

    fn crop(&self, x: usize, y: usize, w: usize, h: usize) -> Self
    {
        let xs = (x..(x+w)).collect::<Vec<_>>();
        let ys = (y..(y+h)).collect::<Vec<_>>();
        Self(self.0.select(Axis(0), ys.as_slice()).select(Axis(1), xs.as_slice()))
    }

    fn hyper_sample(&self, scale: usize) -> Self
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
    fn concat(&self, other: &Self) -> Self
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
    
}


fn range2d(istart: isize, jstart: isize, iend: isize, jend: isize) -> Vec<(isize, isize)>
{
    (jstart..jend).map(move |j| (istart..iend).map(move |i| (i, j)))
        .flatten()
        
        .collect::<Vec<_>>()
}




fn nlmean(img_noisy: &Image, img_ref: &Image) -> Result<Image, Error>
{
    // parameters
    let dist_radius = 10;
    let radius = 2;
    let tau = 0.03;
    let tresh = 0.1;
    
    let diam = 2*radius+1;

    let w = img_noisy.w();
    let h = img_noisy.h();
    let im_len = img_noisy.len();
    let pa_len = diam*diam;

    

    
    
    let patches = img_noisy.patches(radius);
    let mat_patches_shape = (im_len, pa_len);
    let patch_mean = patches.mean_axis(Axis(0)).unwrap();

    let patches_centered = patches - patch_mean.broadcast(mat_patches_shape).unwrap();
    
    let (eig_values, eig_vecs) = pca(patches_centered.clone())?;
    
    let total_val: float = eig_values.iter().sum();
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

    let subspace_raw = vecs.iter().map(|vec| vec.iter()).flatten().cloned().collect::<Vec<float>>();
    let subspace = Array2::from_shape_vec((dim, pa_len), subspace_raw)?;


    //println!("SUBSPACE MATRIX SHAPE: {:?}", subspace.shape());

    let projection = patches_centered.dot(&subspace.t());
 
    
    //println!("Projection (H) shape: {:?}", projection.shape());

    let projection_reshaped = projection.into_shape((h, w, dim))?;
    
    /*
    for i in 0..dim
    {
        let layer = projection_reshaped.select(Axis(2), &[i]).into_shape((h, w))?;
        Image(layer).save(&format!("layer{}.png", i));
    }
     */

    
    let pixel_process = |i0, j0|
    {
        let patch = projection_reshaped.select(Axis(0), &[j0]).select(Axis(1), &[i0]);
        let x0 = i0.max(dist_radius) - dist_radius;
        let y0 = j0.max(dist_radius) - dist_radius;

        let x1 = (i0+dist_radius+1).min(w);
        let y1 = (j0+dist_radius+1).min(h);
        
        let x = (x0..x1).collect::<Vec<_>>();
        let y = (y0..y1).collect::<Vec<_>>();
        let selection = projection_reshaped.select(Axis(0), y.as_slice()).select(Axis(1), x.as_slice())
            - patch.broadcast((y.len(), x.len(), dim)).unwrap();

        let dist = (selection.fold_axis(Axis(2), 0., |sum, x| sum+x*x) / (pa_len as float));
        let prekernel = dist.map(|x| (-x/(2.0*tau*tau)).exp());
        let norm = prekernel.sum();
        let kernel = prekernel/norm;

        let im_selec = img_noisy.0.select(Axis(0), y.as_slice()).select(Axis(1), x.as_slice());
        let pix = (kernel.clone()*im_selec).sum(); 
        (dist, kernel, pix)
        
    };

    if true
    {
        let n_threads = 16;
        let pas = w / n_threads;
        let rest = w % n_threads;

        let indices = (0..n_threads)
            .collect::<Vec<usize>>();
        let mut img_parts = indices
            .par_iter()
            .map(|k|
                 {
                     (
                         k,
                         Image(Array2::from_shape_fn(
                             (h, pas),
                             |(j0, i0)|
                             {
                                 pixel_process(i0+k*pas, j0).2
                             }
                         )))
                 })
            .collect::<Vec<_>>();
        img_parts.sort_by(|(k1, part1), (k2, part2)| k1.cmp(k2));
        let multithreaded = img_parts.iter().fold(Image::empty(0, h), |acc, part| acc.concat(&part.1));

        let multi2 = multithreaded.concat(
            &Image(Array2::from_shape_fn(
                (h, rest),
                |(j0, i0)|
                {
                    pixel_process(i0+w-rest, j0).2
                }
            )));
        Ok(multi2)

    }
    else
    {
        
        let denoized = Image(Array2::from_shape_fn(
            (h, w),
            |(j0, i0)|
            {
                pixel_process(i0, j0).2
            }
        ));

        Ok(denoized)
    }
}



enum Tree
{
    Leaf(Vec<usize>),
    Node(Box<Tree>, Box<Tree>),
}


fn pca(patches_centered: Array2<float>) -> Result<(Vec<float>, Vec<Array1<float>>), Error>
{

    let pa_len = patches_centered.ncols();
    
    let c = patches_centered.t().dot(&patches_centered);
    let (eig_values_complex, eig_vecs_complex) = Eig::eig(&c)?;
    
    let eig_values_real = eig_values_complex.map(|c| c.re);
    let eig_vecs_real = eig_vecs_complex.map(|c| c.re);

    let mut eig_tuples = (0..pa_len).map(|i| (eig_values_real[i], eig_vecs_real.column(i).to_owned())).collect::<Vec<_>>();
    
    eig_tuples.sort_by(|(val1, vec1), (val2, vec2)| val2.partial_cmp(val1).unwrap());

    let tuple: (Vec<_>, Vec<_>) = eig_tuples.into_iter().unzip();
    
    Ok(tuple)
    
}


impl Tree
{
    fn ham_sandwich(patches: &Array2<float>, indices: Vec<usize>, goal: usize) -> Self
    {
        let cut = |patches_indices: Vec<usize>|
        {
            // (u-m)*d
            let selection = patches.select(Axis(0), patches_indices.as_slice());
            let mean = selection.mean_axis(Axis(0)).unwrap();
            let n = patches_indices.len();
            let dim = patches.ncols();
            let centered = selection - mean.broadcast((n, dim)).unwrap();
            let (vals, vecs) = pca(centered.clone()).unwrap();
            let axe = &vecs[0];
//            let dots = centered.dot(&axe.t());

//            println!("DOTs SHAPES {:?}", dots.shape());
            
            let (front, back): (Vec<usize>, Vec<usize>) =
                patches_indices
                .iter()
                .partition(|&&i|
                           {
                               //let dot = dots[[*i]];
                               let dot = (patches.row(i).to_owned() - mean.clone()).dot(axe);
                               dot >= 0.0
                           }
                );
            
            (front, back)
        };


        if indices.len() <= goal
        {
            Tree::Leaf(indices)
        }
        else
        {
            let (front, back) = cut(indices);
            Self::Node(
                Box::new(Self::ham_sandwich(patches, front, goal)),
                Box::new(Self::ham_sandwich(patches, back, goal))
            )
        }
    }

    fn leafs(self) -> Vec<Vec<usize>>
    {
        let mut trees = vec![self];
        let mut leafs = vec![];

        while let Some(tree) = trees.pop()
        {
            match tree
            {
                Tree::Leaf(leaf) => leafs.push(leaf),
                Tree::Node(front, back) => {trees.push(*front); trees.push(*back)}
            }
        }


        leafs
    }
    
}




fn nlmean_ham(img_noisy: &Image, img_ref: &Image) -> Result<Image, Error>
{
    // parameters
    let partitioning = 100;
    let radius = 3;
    let tau = 0.05;
    let tresh = 0.1;
    
    let diam = 2*radius+1;

    let w = img_noisy.w();
    let h = img_noisy.h();
    let im_len = img_noisy.len();
    let pa_len = diam*diam;

    

    
    
    let patches = img_noisy.patches(radius);
    let mat_patches_shape = (im_len, pa_len);
    let patch_mean = patches.mean_axis(Axis(0)).unwrap();

    let patches_centered = patches.clone() - patch_mean.broadcast(mat_patches_shape).unwrap();


    let (eig_values, eig_vecs) = pca(patches_centered.clone())?;
    
    let total_val: float = eig_values.iter().sum();
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
    
    let subspace_raw = vecs.iter().map(|vec| vec.iter()).flatten().cloned().collect::<Vec<float>>();
    let subspace = Array2::from_shape_vec((dim, pa_len), subspace_raw)?;


    //println!("SUBSPACE MATRIX SHAPE: {:?}", subspace.shape());

    let projection = patches_centered.dot(&subspace.t());
 
    
    //println!("Projection (H) shape: {:?}", projection.shape());

    

    
    
    let mut patches_indices = (0..im_len).collect::<Vec<usize>>();

    let tree = Tree::ham_sandwich(&patches, patches_indices, partitioning);

    let leafs = tree.leafs();

    let mut zones = Array1::default((im_len));

    println!("nb of leafs: {}", leafs.len());
    for (zone, leaf) in leafs.iter().enumerate()
    {
        println!("{}", leaf.len());
        for i in leaf
        {
            zones[[*i]] = (zone as float)/(leafs.len() as float);
        }
    }

    Image::from_mat(zones.into_shape((h, w))?).save("zones.png");
    
    

    
    let flat_img = img_noisy.0.clone().into_shape((im_len))?;

    let mut denoised_flat = Array1::default((im_len));
    /*
    for leaf in leafs.iter()
    {

        // selection of the closest looking patches
        let selection = projection.select(Axis(0), leaf.as_slice());
        let im_selec = flat_img.select(Axis(0), leaf.as_slice());

        let sel_len = selection.len();
        for &i in leaf.iter()
        {
            let patch = projection.row(i).to_owned();
            let diff = selection.clone() - patch;
            let dist = (diff.fold_axis(Axis(1), 0., |sum, x| sum + x*x)) / (pa_len as float);
            let prekernel = dist.map(|x| (-x/(2.0*tau*tau)).exp());
            let norm = prekernel.sum();
            let kernel = prekernel/norm;
            
            let pix = (kernel.clone()*im_selec.clone()).sum();
            denoised_flat[[i]] = pix;
        }
    }
     */
    let parts = leafs.par_iter()
        .map(
            |leaf|
            {
                // selection of the closest looking patches
                let selection = projection.select(Axis(0), leaf.as_slice());
                let im_selec = flat_img.select(Axis(0), leaf.as_slice());

                let mut pixels = vec![];
                
                let sel_len = selection.len();
                for &i in leaf.iter()
                {
                    let patch = projection.row(i).to_owned();
                    let diff = selection.clone() - patch;
                    let dist = (diff.fold_axis(Axis(1), 0., |sum, x| sum + x*x)) / (pa_len as float);
                    let prekernel = dist.map(|x| (-x/(2.0*tau*tau)).exp());
                    let norm = prekernel.sum();
                    let kernel = prekernel/norm;
                    
                    let pix = (kernel.clone()*im_selec.clone()).sum();
                    pixels.push(pix);
                }
                pixels
            }).collect::<Vec<_>>();

    for (indices, pixels) in leafs.iter().zip(parts.iter())
    {
        for (i, pix) in indices.iter().zip(pixels.iter())
        {
            denoised_flat[[*i]] = *pix;
        }
    }

    let denoized = Image::from_mat(denoised_flat.into_shape((h, w))?);

    /*
        let denoized = Image(Array2::from_shape_fn(
            (h, w),
            |(j0, i0)|
            {
                pixel_process(i0, j0).2
            }
        ));
*/
        Ok(denoized)
}




fn main() -> Result<(), Error>
{

    let mut sum_snr_denoised = 0.0;
    let mut sum_snr_noisy = 0.0;
/*
    let img_ref = Image::new("images/ricardo_power_post.png")?
        .crop(250, 86, 150, 100)
        ;
     */

    
    let img_ref = Image::new("images/toits.jpg")?
        .crop(128, 100, 120, 128)
        ;

    let noise_level = 0.06;
    let ref_power = img_ref.0.map(|x| x*x).sum().sqrt();
    let theoric_noisy_snr = 20.0*(ref_power/(noise_level*2.0)).log10();

    println!("Theoric snr: {}", theoric_noisy_snr);

    for i in 0..10
    {

        let img_noisy = img_ref.gaussian_noise(noise_level)?;
        let denoized = nlmean_ham(&img_noisy, &img_ref)?;
        
        let snr_denoised = denoized.snr(&img_ref);
        let snr_noisy = img_noisy.snr(&img_ref);

        img_noisy.save(&format!("output/noisy_{:0<3}.png", i));
        denoized.save(&format!("output/denoi_{:0<3}.png", i));
        img_ref.concat(&img_noisy).concat(&denoized).save(&format!("output/all_{:0<3}.png", i));
        sum_snr_denoised += snr_denoised;
        sum_snr_noisy += snr_noisy;
        
        println!("|{:12}|{:12}|{:12}|{:12}|", sum_snr_denoised, sum_snr_noisy, snr_denoised, snr_noisy);
    }

    
    Ok(())
}
