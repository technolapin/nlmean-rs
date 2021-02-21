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




pub struct Chrono
{
    clock: Instant,
    measures: Vec<f64>
}

impl Chrono
{
    pub fn new() -> Self
    {
        Self{clock: Instant::now(), measures: vec![]}
    }

    pub fn record(&mut self)
    {
        self.measures.push(self.clock.elapsed().as_secs_f64());
    }

    pub fn print(&self)
    {
        let mut dur1 = 0.;
        for dur2 in self.measures.iter()
        {
            let dt = dur2-dur1;
            dur1 = *dur2;
            print!(": {:.4}", dt);
        } println!();
    }
    pub fn string(&self) -> String
    {
        let mut s = String::from("|");
        let mut dur1 = 0.;
        for dur2 in self.measures.iter()
        {
            let dt = dur2-dur1;
            dur1 = *dur2;
            s = format!("{} {:.4}|", s, dt);
        }

        s
    }

}


pub type Float = f64;

#[derive(Debug)]
pub struct Error(pub String);


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




pub fn pca(patches_centered: Array2<Float>) -> Result<(Vec<Float>, Vec<Array1<Float>>), Error>
{

    let pa_len = patches_centered.ncols();
    
    let c = patches_centered.t().dot(&patches_centered);
    let (eig_values_complex, eig_vecs_complex) = Eig::eig(&c)?;
    
    let eig_values_real = eig_values_complex.map(|c| c.re);
    let eig_vecs_real = eig_vecs_complex.map(|c| c.re);

    let mut eig_tuples = (0..pa_len).map(|i| (eig_values_real[i], eig_vecs_real.column(i).to_owned())).collect::<Vec<_>>();
    
    eig_tuples.sort_by(|(val1, _vec1), (val2, _vec2)| val2.partial_cmp(val1).unwrap());

    let tuple: (Vec<_>, Vec<_>) = eig_tuples.into_iter().unzip();
    
    Ok(tuple)
    
}



/// used to iterate over 2D coordinates
pub fn range2d(istart: isize, jstart: isize, iend: isize, jend: isize) -> Vec<(isize, isize)>
{
    (jstart..jend).map(move |j| (istart..iend).map(move |i| (i, j)))
        .flatten()
        
        .collect::<Vec<_>>()
}
