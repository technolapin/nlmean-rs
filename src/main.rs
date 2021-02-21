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

use non_local_means::*;


/**
Used to partition the space of patches
The goal is to place similar patches together.
*/
enum Tree
{
    Leaf(Vec<usize>),
    Node(Box<Tree>, Box<Tree>),
}

impl Tree
{

    /**
    tries to separate the centered points from the others.
     */
    fn centered_cut(patches: &Array2<Float>, indices: Vec<usize>) -> (Vec<usize>, Vec<usize>)
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

        (pts_inner.into_iter().cloned().collect(),
         pts_outer.into_iter().cloned().collect())
    }


    /**
    Separates The points by a plane normal to the principal component.
     */
    fn pca_cut(patches: &Array2<Float>, indices: Vec<usize>) -> (Vec<usize>, Vec<usize>)
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
    


    /**
    Separate the points with a good random hyper plane
     */
    fn random_cut(patches: &Array2<Float>, indices: Vec<usize>) -> (Vec<usize>, Vec<usize>)
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
        
        
        
        

        let mut hplanes = vec![];
        for k in 0..n_hplanes
        {
            let points_sample_hp: Vec<Array1<Float>> = indices.choose_multiple(&mut rng, dim)
                .map(|i| patches.row(*i).to_owned().clone())
                .collect();

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
    


    fn ham_cut(patches: &Array2<Float>, indices: Vec<usize>) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>)
    {
        let dim = patches.ncols();

        let mut rng = &mut rand::thread_rng();
        
        let n_hplanes = 10;
     
        let n_points = 1000.min(indices.len());


        let mut hplanes = vec![];
        for k in 0..n_hplanes
        {
            let points_sample_hp: Vec<Array1<Float>> = indices.choose_multiple(&mut rng, dim)
                .map(|i| patches.row(*i).to_owned().clone())
                .collect();

            let norm = hyperplane(&points_sample_hp);
            let pt = points_sample_hp[0].clone();
            
            hplanes.push((pt, norm))
        }
        


        
        let (left, right) = Self::random_cut(patches, indices);

        if left.len().min(right.len()) <= dim
        {
            return (left, right, vec![], vec![]);
        }
        
        
        let points_sample_i_left: Vec<usize> = left.choose_multiple(&mut rng, n_points)
            .cloned()
            .collect();
        let points_sample_i_right: Vec<usize> = right.choose_multiple(&mut rng, n_points)
            .cloned()
            .collect();

        let points_sample_left = patches.select(Axis(0), points_sample_i_left.as_slice());
        let points_sample_right = patches.select(Axis(0), points_sample_i_right.as_slice());

        let n_points_left = points_sample_left.len()/dim;
        let n_points_right = points_sample_right.len()/dim;

        let (best_scores, pt, norm) = hplanes.iter()
            .map(|(pt, norm)|
                 {
                     
                     let score_left = (points_sample_left.clone() - pt.broadcast((n_points_left, dim)).unwrap())
                         .dot(&norm.t()).map(|&x| if x == 0. {0.} else {x/x.abs()}).sum().abs();
                     let score_right = (points_sample_right.clone() - pt.broadcast((n_points_right, dim)).unwrap())
                         .dot(&norm.t()).map(|&x| if x == 0. {0.} else {x/x.abs()}).sum().abs();

                     (score_left + score_right, pt, norm)
                 }
            ).min_by(|a, b|
            {
                let d = a.0-b.0;
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

        let (front_left, back_left): (Vec<usize>, Vec<usize>) =
            left
            .iter()
            .partition(|&&i|
                       {
                           let dot = (patches.row(i).to_owned() - pt.clone()).dot(norm);
                           dot >= 0.0
                       }
            );
        let (front_right, back_right): (Vec<usize>, Vec<usize>) =
            right
            .iter()
            .partition(|&&i|
                       {
                           let dot = (patches.row(i).to_owned() - pt.clone()).dot(norm);
                           dot >= 0.0
                       }
            );
        (front_left, back_left, front_right, back_right)

        
    }


    fn build_pca(patches: &Array2<Float>, indices: Vec<usize>, goal: usize) -> Self
    {


        if indices.len() <= goal
        {
            Tree::Leaf(indices)
        }
        else
        {
            let (front, back) = Self::pca_cut(patches, indices);
            Self::Node(
                Box::new(Self::build_pca(patches, front, goal)),
                Box::new(Self::build_pca(patches, back, goal))
            )
        }
    }


    
    fn build_random(patches: &Array2<Float>, indices: Vec<usize>, goal: usize) -> Self
    {

        if indices.len() <= goal
        {
            Tree::Leaf(indices)
        }
        else
        {
            let (front, back) = Self::random_cut(patches, indices);
            Self::Node(
                Box::new(Self::build_random(patches, front, goal)),
                Box::new(Self::build_random(patches, back, goal))
            )
        }
    }


    
    fn build_sandwich(patches: &Array2<Float>, indices: Vec<usize>, goal: usize) -> Self
    {

        if indices.len() <= goal
        {
            Tree::Leaf(indices)
        }
        else
        {
            let (a, b, c, d) = Self::ham_cut(patches, indices);
            
            Self::Node(
                Box::new(
                    Self::Node(
                        Box::new(Self::build_sandwich(patches, a, goal)),
                        Box::new(Self::build_sandwich(patches, b, goal))
                    )
                ),
                Box::new(
                    Self::Node(
                        Box::new(Self::build_sandwich(patches, c, goal)),
                        Box::new(Self::build_sandwich(patches, d, goal))
                    )
                )
            )

        }
    }


    fn hybrid(patches: &Array2<Float>, indices: Vec<usize>, goal: usize) -> Self
    {
        if indices.len() <= goal
        {
            Self::Leaf(indices)
        }
        else
        {
            let (inner, outer) = Self::centered_cut(patches, indices);
            let (out_for, out_bac) = Self::random_cut(patches, outer);
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
fn hyperplane(points: &Vec<Array1<Float>>) -> Array1<Float>
{
    let d = points.len();
    let dd = points[0].len();
    if d != dd
    {
        panic!(format!("CANNOT BUILD AN HYPERPLANE OF DIM {} WITH {} POINTS", dd, d));
    }
    let mat = Array2::from_shape_fn((d, d), |(j, i)| points[i][[j]] - points[0][[j]]);
    let (_vals, vecs) = pca(mat).unwrap();
    let n = vecs.last().unwrap();
    n.clone()
}




const PATCH_SPACE_PATCHES_RADIUS: usize = 10;
const PATCHES_RADIUS: usize = 2;
const TAU: Float = 0.03;
const PCA_TRESHOLD: Float = 0.1;
const PARTITIONING_GOAL: usize = 500;
const zones_PATCHES_RADIUS: isize = 2;

/**
This is a simple implementation of the NLmean algorithm.
Patches are compared localy in order to avoid the O(n²) complexity of a full-size comparison per pixel.
*/
fn nlmean(img_noisy: &Image) -> Result<Image, Error>
{
    
    let diam = 2*PATCHES_RADIUS+1;

    let w = img_noisy.w();
    let h = img_noisy.h();
    let _im_len = img_noisy.len();
    let pa_len = diam*diam;


    
    let (projection, _patches,  _patch_mean, dim) = img_noisy.projection(PATCHES_RADIUS, PCA_TRESHOLD)?;

    let projection_reshaped = projection.into_shape((h, w, dim))?;
    
    let pixel_process = |i0, j0|
    {
        let patch = projection_reshaped.select(Axis(0), &[j0]).select(Axis(1), &[i0]);
        let x0 = i0.max(PATCH_SPACE_PATCHES_RADIUS) - PATCH_SPACE_PATCHES_RADIUS;
        let y0 = j0.max(PATCH_SPACE_PATCHES_RADIUS) - PATCH_SPACE_PATCHES_RADIUS;

        let x1 = (i0+PATCH_SPACE_PATCHES_RADIUS+1).min(w);
        let y1 = (j0+PATCH_SPACE_PATCHES_RADIUS+1).min(h);
        
        let x = (x0..x1).collect::<Vec<_>>();
        let y = (y0..y1).collect::<Vec<_>>();
        let selection = projection_reshaped.select(Axis(0), y.as_slice()).select(Axis(1), x.as_slice())
            - patch.broadcast((y.len(), x.len(), dim)).unwrap();

        let dist = selection.fold_axis(Axis(2), 0., |sum, x| sum+x*x) / (pa_len as Float);
        let prekernel = dist.map(|x| (-x/(2.0*TAU*TAU)).exp());
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
        img_parts.sort_by(|(k1, _part1), (k2, _part2)| k1.cmp(k2));
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


fn nlmean_tree<F>(img_noisy: &Image, tree_cut: F) -> Result<Image, Error>
where
    F: Fn(&Array2<Float>, Vec<usize>, usize) -> Tree
{
    
    let diam = 2*PATCHES_RADIUS+1;

    let w = img_noisy.w();
    let h = img_noisy.h();
    let im_len = img_noisy.len();
    let pa_len = diam*diam;

        
    let (projection, patches, _patch_mean, _dim) = img_noisy.projection(PATCHES_RADIUS, PCA_TRESHOLD)?;
 
    
    let mut patches_indices = (0..im_len).collect::<Vec<usize>>();

    let tree = tree_cut(&patches, patches_indices, PARTITIONING_GOAL);

    let mut leafs = tree.leafs();

    let mut zones = Array1::default(im_len);

    for (zone, leaf) in leafs.iter().enumerate()
    {
        for i in leaf
        {
            zones[[*i]] = (zone as Float)/(leafs.len() as Float);
        }
    }
    Image::from_mat(zones.into_shape((h, w))?).save("zones.png")?;
    
    let flat_img = img_noisy.0.clone().into_shape(im_len)?;

    let mut denoised_flat = Array1::default(im_len);

    let parts = leafs.par_iter()
        .map(
            |leaf|
            {
                // selection of the closest looking patches
                let selection = projection.select(Axis(0), leaf.as_slice());
                let im_selec = flat_img.select(Axis(0), leaf.as_slice());

                let mut pixels = vec![];
                
                for &i in leaf.iter()
                {
                    let patch = projection.row(i).to_owned();
                    let diff = selection.clone() - patch;
                    let dist = (diff.fold_axis(Axis(1), 0., |sum, x| sum + x*x)) / (pa_len as Float);
                    let prekernel = dist.map(|x| (-x/(2.0*TAU*TAU)).exp());
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

    Ok(denoized)
}


fn nlmean_ham_fuse(img_noisy: &Image) -> Result<Image, Error>
{
    let diam = 2*PATCHES_RADIUS+1;

    let w = img_noisy.w();
    let h = img_noisy.h();
    let im_len = img_noisy.len();
    let pa_len = diam*diam;
    
    let mut chrono = Chrono::new();
    
    
    let (projection, patches, _patch_mean, dim) = img_noisy.projection(PATCHES_RADIUS, PCA_TRESHOLD)?;

    
    let gauss = |x: &Float| (-x/(2.0*TAU*TAU)).exp();
    
    let mut patches_indices = (0..im_len).collect::<Vec<usize>>();


    chrono.record();

    let tree = Tree::build_pca(&patches, patches_indices, PARTITIONING_GOAL);

    let mut leafs = tree.leafs();


    chrono.record();


    
    let mut zones = Array1::default(im_len);

    let nl = leafs.len();
    



    let mut rng = rand::thread_rng();

    
    let mut approx_mean = |i: usize|
    {
        let leaf: &Vec<usize> = &leafs[i];
        let n_samples = 10;
        (0..n_samples).map(|_| projection.row(leaf[rng.gen_range(0..leaf.len())]))
            .fold(Array1::<Float>::default(dim), |sum, p| sum+p)/(n_samples as Float)

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
    for (zone, leaf) in leafs.iter().enumerate()
    {
        for i in leaf
        {
            zones[[*i]] = (zone as Float)/(leafs.len() as Float);
        }
    }
    Image::from_mat(zones.into_shape((h, w))?).save("zones_fuse.png")?;


   
    
    let flat_img = img_noisy.0.clone().into_shape(im_len)?;

    let mut denoised_flat = Array1::default(im_len);

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
                 
                 for &i in leaf.iter()
                 {
                     let patch = projection.row(i).to_owned();
                     let diff = selection.clone() - patch;
                     let dist = (diff.fold_axis(Axis(1), 0., |sum, x| sum + x*x)) / (pa_len as Float);
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

    Ok(denoized)
}




fn nlmean_ham_local(img_noisy: &Image) -> Result<Image, Error>
{    
    let diam = 2*PATCHES_RADIUS+1;

    let w = img_noisy.w();
    let h = img_noisy.h();
    let im_len = img_noisy.len();
    let pa_len = diam*diam;

        
    
    let (projection, patches, patch_mean, dim) = img_noisy.projection(PATCHES_RADIUS, PCA_TRESHOLD)?;

    
    
    let mut patches_indices = (0..im_len).collect::<Vec<usize>>();

    let tree = Tree::build_pca(&patches, patches_indices, PARTITIONING_GOAL);

    let mut leafs = tree.leafs();

    let mut zones = Array1::default(im_len);

    for (zone, leaf) in leafs.iter().enumerate()
    {
        for i in leaf
        {
            zones[[*i]] = zone;
        }
    }
    Image::from_mat(zones.map(|&zone| (zone as Float)/(leafs.len() as Float))
                    .into_shape((h, w))?)
        .save("zones.png")?;

    let mut rng = rand::thread_rng();

    let mut approx_mean = |i: usize|
    {
        let leaf: &Vec<usize> = &leafs[i];
        let n_samples = 10;
        (0..n_samples).map(|_| projection.row(leaf[rng.gen_range(0..leaf.len())]))
            .fold(Array1::<Float>::default(dim), |sum, p| sum+p)/(n_samples as Float)
            
    };
    let means = (0..leafs.len()).map(|i| approx_mean(i)).collect::<Vec<_>>();

    
    let flat_img = img_noisy.0.clone().into_shape(im_len)?;

    let mut denoised_flat = Array1::default(im_len);

    let mut get_zones = |i: usize|
    {
        let x = (i % w) as isize;
        let y = (i / w) as isize;
        range2d(x-zones_PATCHES_RADIUS, y-zones_PATCHES_RADIUS, x+1+zones_PATCHES_RADIUS, y+1+zones_PATCHES_RADIUS)
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

        Image(test).save(&format!("test/teeeest{:0<5}.png", k))?;
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
                    let dist = (diff.fold_axis(Axis(1), 0., |sum, x| sum + x*x)) / (pa_len as Float);
                    let prekernel = dist.map(|x| (-x/(2.0*TAU*TAU)).exp());
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

    Ok(denoized)
}



fn main() -> Result<(), Error>
{
/*
    let img_ref = Image::new("images/ricardo_power_post.png")?
        .crop(250, 86, 150, 100)
        ;
     */

    
    let img_ref = Image::new("images/toits.jpg")?
        .crop(128, 100, 120, 128)
//          .crop(128, 100, 300, 300)
        ;

    let noise_level = 0.1;
    let ref_power = img_ref.0.map(|x| x*x).sum().sqrt();
    let theoric_noisy_snr = 20.0*(ref_power/(noise_level*2.0)).log10();

    println!("Theoric snr: {}", theoric_noisy_snr);

    let steps = 10;
    let mut chronos = vec![];

    let mut all_outputs = Image::empty(6*img_ref.w(), 0);
        
    for i in 1..=steps
    {
        let mut chrono = Chrono::new();

        let img_noisy = img_ref.gaussian_noise(noise_level*(i as Float)/(steps as Float))?;
        let snr_noisy = img_noisy.snr(&img_ref);

        let den1 = nlmean(&img_noisy)?;
        chrono.record();
        let den2 = nlmean_tree(&img_noisy, Tree::build_pca)?;
        chrono.record();
        let den22 = nlmean_tree(&img_noisy, Tree::hybrid)?;
        chrono.record();
        let den3 = nlmean_tree(&img_noisy, Tree::build_sandwich)?;
        chrono.record();

        chronos.push(chrono.string());
//        chrono.print();
        
        let denoizeds = vec![
            den1, den2, den22, den3
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

        all_outputs = all_outputs.concat_vertical(&output);

        output.save(&format!("output/all_{:0<3}.png", i))?;


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

    println!("DIIMSSS {} {}", all_outputs.w(), all_outputs.h());
    
    all_outputs.save("output/all_all.png")?;
    
    for times in chronos
    {
        println!("{}", times);
    }
    
    Ok(())
}
