use image::{ImageBuffer};
use image::{Luma};

use ndarray::{Array1, Array2, Array3, Axis};
use rand_distr::{Distribution, Normal};
use ndarray_linalg::*;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use rand::{thread_rng, Rng};
use rand::prelude::SliceRandom;


type float = f64;

#[derive(Debug)]
struct Error(String);
use std::cmp::Ordering;


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

    fn projection(&self, radius: usize, tresh: float) -> Result<(Array2<float>, Array2<float>, Array1<float>, usize), Error>
    {
        let im_len = self.len();
        let pa_len = (radius*2+1).pow(2);
        
        let patches = self.patches(radius);
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



        let projection = patches_centered.dot(&subspace.t());

        Ok((projection, patches, patch_mean, dim))
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

    /// used to augment the resolution of an image in order
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


    
    let (projection, _patches,  patch_mean, dim) = img_noisy.projection(radius, tresh)?;

    let projection_reshaped = projection.into_shape((h, w, dim))?;
    
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

    // parallelizing the computation
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



enum Tree
{
    Leaf(Vec<usize>),
    Node(Box<Tree>, Box<Tree>),
}




impl Tree
{

    fn centered_cut(patches: &Array2<float>, indices: Vec<usize>) -> (Vec<usize>, Vec<usize>)
    {
        let mut norms = indices.iter().map(|&i| (i, patches.row(i).to_owned().map(|x| x*x).sum().sqrt())).collect::<Vec<_>>();
        norms.sort_by(|a, b|
                      {
                          let d = a.1-b.1;
                          if d < 0.
                          {
                              Ordering::Less
                          }
                          else if d > 0.
                          {
                              Ordering::Greater
                          }
                          else
                          {
                              Ordering::Equal
                          }
                      });
        let max = norms.iter()
            .max_by(|a, b|
                    {
                        let d = a.1-b.1;
                        if d < 0.
                        {
                            Ordering::Less
                        }
                        else if d > 0.
                        {
                            Ordering::Greater
                        }
                        else
                        {
                            Ordering::Equal
                        }
                    }
            ).unwrap();
        let histo_res = max.1/60.;
        let mut histogram = vec![0; (max.1/histo_res) as usize];
        let mut overflow = 0;

        for (i, n) in norms
        {
            let bin = (n/histo_res) as usize;
            if bin >= histogram.len()
            {
                overflow+=1;
            }
            else
            {
                histogram[bin]+=1;
            }
        }

        let mut first_valley = histogram.len();
        let mut sum = histogram[0];
        for i in 1..(histogram.len()-1)
        {
            let a = histogram[i-1];
            let b = histogram[i];
            let c = histogram[i+1];
            sum += b;
            if a > b && b < c
            {
                first_valley = i;
                break;
            }
            
        }


        let pts_inner = &indices[..sum];
        let pts_outer = &indices[sum..];


        
        /*
        println!("valley: {}", first_valley);
        
        print!("CENTERED HISTOGRAM NORMS (overflow: {})\n", overflow);

        for i in 0..histogram.len()
        {
            print!("{:.1} ", (i as float)*histo_res);
        } println!();
        for bin in histogram
        {
            print!("{:>3} ", bin);
        } println!();
         */

        
        (pts_inner.into_iter().cloned().collect(),
         pts_outer.into_iter().cloned().collect())

        
    }

    fn pca_cut(patches: &Array2<float>, indices: Vec<usize>) -> (Vec<usize>, Vec<usize>)
    {
            let selection = patches.select(Axis(0), indices.as_slice());
            let mean = selection.mean_axis(Axis(0)).unwrap();
            let n = indices.len();
            let dim = patches.ncols();
            let centered = selection - mean.broadcast((n, dim)).unwrap();
            let (vals, vecs) = pca(centered.clone()).unwrap();
            let axe = &vecs[0];
            
            let (front, back): (Vec<usize>, Vec<usize>) =
                indices
                .iter()
                .partition(|&&i|
                           {
                               let dot = (patches.row(i).to_owned() - mean.clone()).dot(axe);
                               dot >= 0.0
                           }
                );
            
            (front, back)
        
    }
    



    fn ham_cut(patches: &Array2<float>, indices: Vec<usize>) -> (Vec<usize>, Vec<usize>)
    {
        let n_points = 100.min(indices.len());
        let dim = patches.ncols();

        // to avoid the cases where we do not have enough points to build hyperplanes
        if n_points <= dim
        {
            return (indices, vec![]);
        }
        
        let n_hplanes = 10; 
        let mut rng = &mut rand::thread_rng();
        
        let points_sample_i: Vec<usize> = indices.choose_multiple(&mut rng, n_points)
            .cloned()
            .collect();

        let points_sample = patches.select(Axis(0), points_sample_i.as_slice());
        
        
        
        
  //      println!("Sampling sets of {} points", dim);

        let mut hplanes = vec![];
        for k in 0..n_hplanes
        {
            let points_sample_hp: Vec<Array1<float>> = indices.choose_multiple(&mut rng, dim)
                .map(|i| patches.row(*i).to_owned().clone())
                .collect();

//            println!("* Sampled {} points from {}", points_sample_hp.len(), indices.len());
            let norm = hyperplane(&points_sample_hp);
            let pt = points_sample_hp[0].clone();

            let score = (points_sample.clone() - pt.broadcast((n_points, dim)).unwrap())
                .dot(&norm.t()).map(|&x| if x == 0. {0.} else {x/x.abs()}).sum().abs();
            
            hplanes.push((pt, norm, score))
        }
        // (u-m)*d
        let (pt, norm, best_score) = hplanes.iter().min_by(
            |a, b|
            {
                let d = a.2-b.2;
                if d < 0.
                {
                    Ordering::Less
                }
                else if d > 0.
                {
                    Ordering::Greater
                }
                else
                {
                    Ordering::Equal
                }
            }                                               
        ).unwrap();
        
        let (front, back): (Vec<usize>, Vec<usize>) =
            indices
            .iter()
            .partition(|&&i|
                       {
                           let dot = (patches.row(i).to_owned() - pt.clone()).dot(norm);
                           dot >= 0.0
                       }
            );
        
        (front, back)
    }
    


    fn ham_sandwich(patches: &Array2<float>, indices: Vec<usize>, goal: usize) -> Self
    {


        if indices.len() <= goal
        {
            Tree::Leaf(indices)
        }
        else
        {
            let (front, back) = Self::pca_cut(patches, indices);
            Self::Node(
                Box::new(Self::ham_sandwich(patches, front, goal)),
                Box::new(Self::ham_sandwich(patches, back, goal))
            )
        }
    }

    
    fn ham_sandwich2(patches: &Array2<float>, indices: Vec<usize>, goal: usize) -> Self
    {

        if indices.len() <= goal
        {
            Tree::Leaf(indices)
        }
        else
        {
            let (front, back) = Self::ham_cut(patches, indices);
            Self::Node(
                Box::new(Self::ham_sandwich2(patches, front, goal)),
                Box::new(Self::ham_sandwich2(patches, back, goal))
            )
        }
    }

    fn hybrid(patches: &Array2<float>, indices: Vec<usize>, goal: usize) -> Self
    {
        if indices.len() <= goal
        {
            Self::Leaf(indices)
        }
        else
        {
            let (inner, outer) = Self::centered_cut(patches, indices);
            let (out_for, out_bac) = Self::ham_cut(patches, outer);
            Self::Node(
                Box::new(Self::Leaf(inner)),
                Box::new(
                    Self::Node(
                        Box::new(Self::hybrid(patches, out_for, goal)),
                        Box::new(Self::hybrid(patches, out_bac, goal))
                        )
                    )
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

/// returns the normal of the hyperplane passing through the points given
fn hyperplane(points: &Vec<Array1<float>>) -> Array1<float>
{
    let d = points.len();
    let dd = points[0].len();
    if d != dd
    {
        panic!(format!("CANNOT BUILD AN HYPERPLANE OF DIM {} WITH {} POINTS", dd, d));
    }
    let mat = Array2::from_shape_fn((d, d), |(j, i)| points[i][[j]] - points[0][[j]]);
    let (vals, vecs) = pca(mat).unwrap();
    let n = vecs.last().unwrap();
    n.clone()
}



fn nlmean_ham<F>(img_noisy: &Image, img_ref: &Image, tree_cut: F) -> Result<Image, Error>
where
    F: Fn(&Array2<float>, Vec<usize>, usize) -> Tree
{
    // parameters
    let partitioning = 500;
    let radius = 3;
    let tau = 0.05;
    let tresh = 0.1;
    
    let diam = 2*radius+1;

    let w = img_noisy.w();
    let h = img_noisy.h();
    let im_len = img_noisy.len();
    let pa_len = diam*diam;

        
    let (projection, patches, patch_mean, dim) = img_noisy.projection(radius, tresh)?;
 
    
    let mut patches_indices = (0..im_len).collect::<Vec<usize>>();

    let tree = tree_cut(&patches, patches_indices, partitioning);

    let mut leafs = tree.leafs();

    let mut zones = Array1::default((im_len));

//    println!("nb of leafs: {}", leafs.len());
    for (zone, leaf) in leafs.iter().enumerate()
    {
        //println!("{}", leaf.len());
        for i in leaf
        {
            zones[[*i]] = (zone as float)/(leafs.len() as float);
        }
    }
    Image::from_mat(zones.into_shape((h, w))?).save("zones.png");

    let mut rng = rand::thread_rng();

    let mut approx_mean = |i: usize|
    {
        let leaf: &Vec<usize> = &leafs[i];
        let n_samples = 10;
        (0..n_samples).map(|_| projection.row(leaf[rng.gen_range(0..leaf.len())]))
            .fold(Array1::<float>::default((dim)), |sum, p| sum+p)/(n_samples as float)

    };
    
    let flat_img = img_noisy.0.clone().into_shape((im_len))?;

    let mut denoised_flat = Array1::default((im_len));

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


fn nlmean_ham_fuse(img_noisy: &Image, img_ref: &Image) -> Result<Image, Error>
{
    // parameters
    let radius = 3;
    let tau = 0.05;
    let tresh = 0.1;
    
    let diam = 2*radius+1;

    let w = img_noisy.w();
    let h = img_noisy.h();
    let im_len = img_noisy.len();
    let pa_len = diam*diam;

    let partitioning = 500;

    let mut chrono = Chrono::new();
    
    
    let (projection, patches, patch_mean, dim) = img_noisy.projection(radius, tresh)?;

    
    let gauss = |x: &float| (-x/(2.0*tau*tau)).exp();
    
    let mut patches_indices = (0..im_len).collect::<Vec<usize>>();


    chrono.record();

    let tree = Tree::ham_sandwich(&patches, patches_indices, partitioning);

    let mut leafs = tree.leafs();


    chrono.record();


    
    let mut zones = Array1::default((im_len));

    let nl = leafs.len();
    



    let mut rng = rand::thread_rng();

    
    let mut approx_mean = |i: usize|
    {
        let leaf: &Vec<usize> = &leafs[i];
        let n_samples = 10;
        (0..n_samples).map(|_| projection.row(leaf[rng.gen_range(0..leaf.len())]))
            .fold(Array1::<float>::default((dim)), |sum, p| sum+p)/(n_samples as float)

    };

    
    let mut dist = |i, j|
    {
        (approx_mean(i) - approx_mean(j)).map(|x| x*x).sum().sqrt()
    };


    let mut fused = (0..nl)
        .map(|i| vec![i])
        .collect::<Vec<_>>();

    let histo_res = 0.1;
    let mut histogram = vec![0; (3.0/histo_res) as usize];
    let mut overflow = 0;
    for i in 0..leafs.len()
    {
        for j in (i+1)..leafs.len()
        {
            let d = dist(i, j);
            //println!("DISTS: {}", d);
            let bin = (d/histo_res) as usize;
            if bin >= histogram.len()
            {
                overflow+=1;
            }
            else
            {
                histogram[bin]+=1;
            }
            if d < 0.3
            {
                fused[i].push(j);
                fused[j].push(i);
            }
        }
    }
/*    
    for i in 0..histogram.len()
    {
        print!("{:.1} ", (i as float)*histo_res);
    } println!();
    for bin in histogram
    {
        print!("{:>3} ", bin);
    } println!();

    
    println!("nb of leafs: {}", leafs.len());
*/
    for (zone, leaf) in leafs.iter().enumerate()
    {
        for i in leaf
        {
            zones[[*i]] = (zone as float)/(leafs.len() as float);
        }
    }
    Image::from_mat(zones.into_shape((h, w))?).save("zones_fuse.png");


   
    
    let flat_img = img_noisy.0.clone().into_shape((im_len))?;

    let mut denoised_flat = Array1::default((im_len));

    let decimating = 10;
    let sampled_leaf =
        leafs.iter()
        .map(|leaf|
             {
                 let n = leaf.len()/decimating;
                 (0..n).map(|_| rng.gen_range(0..leaf.len()))
                     .map(|i| leaf[i])
                     .collect::<Vec<_>>()
             }
        ).collect::<Vec<_>>();

    
    let fused_leafs = fused.iter()
        .map(|i_leafs| i_leafs.iter()
             .map(|&i| leafs[i].iter().cloned())
             .flatten().collect::<Vec<_>>())
        .collect::<Vec<_>>();

//    println!("FUSED {:?}", fused_leafs.iter().map(|v| v.len()).collect::<Vec<_>>());
    
    let parts = (0..leafs.len()).collect::<Vec<_>>()
        .par_iter()
        .map(|&j|
             {
                 let leaf = &leafs[j];
                 let leaf_sample = &sampled_leaf[j];
                 // selection of the closest looking patches
                 let selection = projection.select(Axis(0), leaf_sample.as_slice());
                 let im_selec = flat_img.select(Axis(0), leaf_sample.as_slice());
                 
                 let mut pixels = vec![];
                 
                 let sel_len = selection.len();
                 for &i in leaf.iter()
                 {
                     let patch = projection.row(i).to_owned();
                     let diff = selection.clone() - patch;
                     let dist = (diff.fold_axis(Axis(1), 0., |sum, x| sum + x*x)) / (pa_len as float);
                     let prekernel = dist.map(|x| gauss(x));
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















fn nlmean_ham_local(img_noisy: &Image, img_ref: &Image) -> Result<Image, Error>
{
    // parameters
    let partitioning = 50;
    let radius = 3;
    let tau = 0.05;
    let tresh = 0.002;
    let zones_radius = 1;
    
    let diam = 2*radius+1;

    let w = img_noisy.w();
    let h = img_noisy.h();
    let im_len = img_noisy.len();
    let pa_len = diam*diam;

        
    
    let (projection, patches, patch_mean, dim) = img_noisy.projection(radius, tresh)?;

    
    
    let mut patches_indices = (0..im_len).collect::<Vec<usize>>();

    let tree = Tree::ham_sandwich(&patches, patches_indices, partitioning);

    let mut leafs = tree.leafs();

    let mut zones = Array1::default((im_len));

//    println!("nb of leafs: {}", leafs.len());
    for (zone, leaf) in leafs.iter().enumerate()
    {
        //println!("{}", leaf.len());
        for i in leaf
        {
            zones[[*i]] = zone;
        }
    }
    Image::from_mat(zones.map(|&zone| (zone as float)/(leafs.len() as float)).into_shape((h, w))?).save("zones.png");

    let mut rng = rand::thread_rng();

    let mut approx_mean = |i: usize|
    {
        let leaf: &Vec<usize> = &leafs[i];
        let n_samples = 10;
        (0..n_samples).map(|_| projection.row(leaf[rng.gen_range(0..leaf.len())]))
            .fold(Array1::<float>::default((dim)), |sum, p| sum+p)/(n_samples as float)

    };
    let means = (0..leafs.len()).map(|i| approx_mean(i)).collect::<Vec<_>>();

    
    let flat_img = img_noisy.0.clone().into_shape((im_len))?;

    let mut denoised_flat = Array1::default((im_len));

    let mut get_zones = |i: usize|
    {
        let x = (i % w) as isize;
        let y = (i / w) as isize;
        range2d(x-zones_radius, y-zones_radius, x+1+zones_radius, y+1+zones_radius)
            .iter()
            .map(|&(x, y)| x as usize + (y as usize*w))
            .filter_map(|i| zones.get([i]))
            .cloned()
            .collect::<Vec<usize>>()
    };

    let ngh_leafs = (0..leafs.len())
        .map(|i|
             {
                 let mut v = vec![false; leafs.len()];
                 let zns: Vec<_> = leafs.iter().map(|j| get_zones(i)).collect();
                 v[i] = true;
                 for &ngh in zns.iter().map(|z| z.iter()).flatten()
                 {
                     if (means[i].clone() - means[ngh].clone()).map(|x| x*x).sum().sqrt() < 0.3
                     {
                         v[ngh] = true;
                     }
                 }
                 v.iter()
                     .enumerate()
                     .filter_map(|(i, b)| if *b {Some(i)} else {None})
                     .collect::<Vec<_>>()
             })
        .collect::<Vec<_>>();


    for k in 0..ngh_leafs.len()
    {
        let mut test = Array2::default((h, w));
        for &i in ngh_leafs[k].iter()
        {
            for j in &leafs[i]
            {
                test[[j/w, j%w]] = 1.0;
            }
        }

        Image(test).save(&format!("test/teeeest{:0<5}.png", k));
    }
    
    
    
    let parts = (0..leafs.len()).collect::<Vec<_>>()
        .par_iter()
        .map(
            |&j|
            {
                
                let zones = get_zones(j);
                //let leaf = & leafs[i];

                let leaf = ngh_leafs[j]
                    .iter()
                    .map(|&j| leafs[j].iter())
                    .flatten()
                    .cloned()
                    .collect::<Vec<_>>();
                
                let mut pixels = vec![];
                for &i in leaf.iter()
                {
                    // selection of the closest looking patches
                    let selection = projection.select(Axis(0), leaf.as_slice());
                    let im_selec = flat_img.select(Axis(0), leaf.as_slice());
                    
                    let sel_len = selection.len();


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





/*
fn nlmean_ham2(img_noisy: &Image, img_ref: &Image) -> Result<Image, Error>
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

    

    let mut rng = rand::thread_rng();
    
    
    let mut patches_indices = (0..im_len).collect::<Vec<usize>>();

    let tree = Tree::ham_sandwich2(&patches, patches_indices, partitioning);

    let mut leafs = tree.leafs();

    // I only sample one point for now (the first one)
    let sample_i: Vec<usize> = leafs.iter()
        .map(|leaf| leaf[0]).collect();

    let sample = projection.select(Axis(0), sample_i.as_slice());

    
    
    
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


    let mut approx_mean = |i: usize|
    {
        let leaf: &Vec<usize> = &leafs[i];
        let n_samples = 10;
        (0..n_samples).map(|_| projection.row(leaf[rng.gen_range(0..leaf.len())]))
            .fold(Array1::<float>::default((dim)), |sum, p| sum+p)/(n_samples as float)

    };
    if false
    {
        let means = (0..leafs.len()).map(|i| approx_mean(i)).collect::<Vec<_>>();

        let proximities = Array2::from_shape_fn((leafs.len(), leafs.len()),
                                                |(j, i)|
                                                {
                                                    (means[j].clone()-means[i].clone()).map(|x| x*x).sum()
                                                }
        );

        println!("{:?}", proximities);
        Image::from_mat(proximities).save("prox.png");

        let min_dist = 0.7;

        let mut fusions = (0..leafs.len()).map(|i| vec![i]).collect::<Vec<_>>();

        for i in 0..leafs.len()
        {
            if fusions[i].is_empty()
            {
                continue;
            }
            for j in (i+1)..leafs.len()
            {
                let dist2 = (means[j].clone()-means[i].clone()).map(|x| x*x).sum();
                if dist2 < min_dist
                {
                    fusions[j].clear();
                    fusions[i].push(j);
                }
            }
        }

        let mut leaf_fu = vec![];
        for fu in fusions.iter().filter(|f| !f.is_empty())
        {
            let mut v = vec![];
            for i in fu.iter()
            {
                v.append(&mut leafs[*i]);
            }
            leaf_fu.push(v);
            
        }
        leafs = leaf_fu;
        


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
        Image::from_mat(zones.into_shape((h, w))?).save("zones2.png");
    }
    

    
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


*/


use std::time::{Instant, Duration};
struct Chrono
{
    clock: Instant,
    measures: Vec<f64>
}

impl Chrono
{
    fn new() -> Self
    {
        Self{clock: Instant::now(), measures: vec![]}
    }

    fn record(&mut self)
    {
        self.measures.push(self.clock.elapsed().as_secs_f64());
    }

    fn print(&self)
    {
        let mut dur1 = 0.;
        for dur2 in self.measures.iter()
        {
            let dt = dur2-dur1;
            dur1 = *dur2;
            print!(": {:.4}", dt);
        } println!();
    }
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

    let noise_level = 0.02;
    let ref_power = img_ref.0.map(|x| x*x).sum().sqrt();
    let theoric_noisy_snr = 20.0*(ref_power/(noise_level*2.0)).log10();

    println!("Theoric snr: {}", theoric_noisy_snr);

    for i in 0..10
    {
        let mut chrono = Chrono::new();

        let img_noisy = img_ref.gaussian_noise(noise_level)?;
        let snr_noisy = img_noisy.snr(&img_ref);

        let den1 = nlmean(&img_noisy, &img_ref)?;
        chrono.record();
        let den2 = nlmean_ham(&img_noisy, &img_ref, Tree::ham_sandwich)?;
        chrono.record();
        let den22 = nlmean_ham(&img_noisy, &img_ref, Tree::hybrid)?;
        chrono.record();

        chrono.print();
        
        let denoizeds = vec![
            den1, den2, den22
         //   nlmean_ham2(&img_noisy, &img_ref)?,
        ];

        let snr_denoizeds = denoizeds.iter()
            .map(|den| den.snr(&img_ref))
            .collect::<Vec<_>>();

        let output = denoizeds.iter()
            .fold(
            img_ref.concat(&img_noisy),
            |out, den|
            {
                out.concat(den)
            }
        );


        output.save(&format!("output/all_{:0<3}.png", i));

        print!("|{:12}|", snr_noisy);
        for den_snr in snr_denoizeds.iter()
        {
            print!("{:12}|", den_snr);
        }
        println!();

            
            
//        sum_snr_denoised += snr_denoised;
  //      sum_snr_noisy += snr_noisy;
        
        //   println!("|{:12}|{:12}|{:12}|{:12}|", sum_snr_denoised, sum_snr_noisy, snr_denoised, snr_noisy);

            
    }

    
    Ok(())
}
