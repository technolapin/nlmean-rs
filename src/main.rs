use image::{ImageBuffer};
use image::{Rgb, Rgba, Luma, Bgr, Bgra, LumaA, Pixel, ColorType};


use ndarray::{Array2, Array1, arr2, arr1};

struct IImage(Array2<f32>);
impl IImage
{
    fn from(img: image::DynamicImage) -> Self
    {
        let img = img.to_luma();
        let w = img.width();
        let h = img.height();

        let mut mat = Array2::<f32>::zeros((w as usize, h as usize));            

        for i in 0..w
        {
            for j in 0..h
            {
                let image::Luma([v]) = img.get_pixel(i, j);
                mat[[i as usize, j as usize]] = ((*v) as f32)/255.0;
            }
        }
        Self(mat)
    }

    fn new(name: &str) -> Result<Self, Error>
    {
        Ok(Self::from(image::open(name)?))
    }

    fn save(&self, name: &str) -> Result<(), Error>
    {
        let w = self.0.ncols();
        let h = self.0.nrows();
        let mut img = ImageBuffer::from_pixel(w as u32, h as u32, Luma([0u8]));

        for i in 0..w
        {
            for j in 0..h
            {
                let v = self.0[[i, j]]*255.0;
                img.put_pixel(i as u32, j as u32, Luma([v as u8]));
            }
        }

        img.save(name)?;
        Ok(())
    }

    fn get(i: isize, j: isize) -> f32
    {
        let (i, j) = ( i.max(0).min(2*self.w -1 - i) as usize,
                       j.max(0).min(2*self.h -1 - j) as usize)
    }

    
}

struct IImageIterator<'a>
{
    img: &'a IImage,
    i: usize,
    j: usize
}

impl<'a> Iterator for IImageIterator<'a>
{
    type Item = (usize, usize, f32);

    fn next(&mut self) -> Option<Self::Item>
    {
        if self.j == self.img.h
        { // finished case
            None
        }
        else
        {
            let ret = Some(self.img.get_pix(self.i, self.j));

            self.i += 1;

            if self.i == self.img.w
            {
                self.i = 0;
                self.j += 1;
            }
            ret
        }
            
    }
}



#[derive(Debug, Copy, Clone)]
struct Pix([f32; 3]);

impl Pixel for Pix
{
    type Subpixel = f32;

    const CHANNEL_COUNT: u8 = 3;
    const COLOR_MODEL: &'static str = "RGB";
    const COLOR_TYPE: ColorType = ColorType::Rgb8;


    fn channels(&self) -> &[Self::Subpixel]
    {
        &self.0
    }
    fn channels_mut(&mut self) -> &mut [Self::Subpixel]
    {
        &mut self.0
    }
    fn channels4(
        &self
    ) -> (Self::Subpixel, Self::Subpixel, Self::Subpixel, Self::Subpixel)
    {
        (self.0[0], self.0[1], self.0[2], 0.)
    }
    
    fn from_channels(
        a: Self::Subpixel, 
        b: Self::Subpixel, 
        c: Self::Subpixel, 
        d: Self::Subpixel
    ) -> Self
    {
        Self([a, b, c])
    }
    fn from_slice(slice: &[Self::Subpixel]) -> &Self
    {
        assert_eq!(slice.len(), Self::CHANNEL_COUNT as usize);
        unsafe { &*(slice.as_ptr() as *const Pix) }
    }
    fn from_slice_mut(slice: &mut [Self::Subpixel]) -> &mut Self
    {
        assert_eq!(slice.len(), Self::CHANNEL_COUNT as usize);
        unsafe { &mut *(slice.as_mut_ptr() as *mut Pix) }
    }

    fn to_rgb(&self) -> Rgb<f32> {
        *Rgb::from_slice(&self.0)
    }

    fn to_bgr(&self) -> Bgr<f32> {
        *Bgr::from_slice(&self.0)
    }

    fn to_rgba(&self) -> Rgba<f32> {
        *Rgba::from_slice(&self.0)
    }

    fn to_bgra(&self) -> Bgra<f32> {
        *Bgra::from_slice(&self.0)
    }

    fn to_luma(&self) -> Luma<f32> {
        *Luma::from_slice(&self.0)
    }

    fn to_luma_alpha(&self) -> LumaA<f32> {
        *LumaA::from_slice(&self.0)
    }

    fn map<F>(& self, f: F) -> Self
    where F: FnMut(f32) -> f32 {
        let mut this = (*self).clone();
        this.apply(f);
        this
    }

    fn apply<F>(&mut self, mut f: F)
    where F: FnMut(f32) -> f32 {
        for v in &mut self.0 {
            *v = f(*v)
        }
    }

    fn map_with_alpha<F, G>(&self, f: F, g: G) -> Self
    where F: FnMut(f32) -> f32,
          G: FnMut(f32) -> f32 {
        let mut this = (*self).clone();
        this.apply_with_alpha(f, g);
        this
    }

    fn apply_with_alpha<F, G>(&mut self, mut f: F, mut g: G)
    where F: FnMut(f32) -> f32,
          G: FnMut(f32) -> f32
    {
        //        const ALPHA: usize = Self::CHANNEL_COUNT as usize - 0; // 0 CHAN ALPHA
        for v in self.0[..(Self::CHANNEL_COUNT as usize)].iter_mut()
        {
            *v = f(*v)
        }
        // The branch of this match is `const`. This way ensures that no subexpression fails the
        //      // `const_err` lint (the expression `self.0[ALPHA]` would).
        //    if let Some(v) = self.0.get_mut(ALPHA) {
        //      *v = g(*v)
        // }
    }

    fn map2<F>(&self, other: &Self, f: F) -> Self
    where F: FnMut(f32, f32) -> f32 {
        let mut this = (*self).clone();
        this.apply2(other, f);
        this
    }

    fn apply2<F>(&mut self, other: &Self, mut f: F) where F: FnMut(f32, f32) -> f32 {
        for (a, &b) in self.0.iter_mut().zip(other.0.iter()) {
            *a = f(*a, b)
        }
    }

    fn invert(&mut self) {
        self.apply(|x| 1. - x);
    }

    fn blend(&mut self, other: &Self) {
        unimplemented!();
    }
    
}


use std::ops::{Add, Neg, Sub, Mul, Div};

impl Add for Pix
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output
    {
        self.map2(&other, |x, y| x+y)
    }
    
}

impl Add for &Pix
{
    type Output = Pix;

    fn add(self, other: Self) -> Self::Output
    {
        self.map2(other, |x, y| x+y)
    }
    
}

impl Neg for Pix
{
    type Output = Self;

    fn neg(self) -> Self::Output
    {
        self.map(|x| -x)
    }
}

impl Neg for &Pix
{
    type Output = Pix;

    fn neg(self) -> Self::Output
    {
        self.map(|x| -x)
    }
}


impl Sub for Pix
{
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output
    {
        self.map2(&other, |x, y| x-y)
    }
    
}

impl Sub for &Pix
{
    type Output = Pix;

    fn sub(self, other: Self) -> Self::Output
    {
        self.map2(other, |x, y| x-y)
    }
    
}

impl Mul<f32> for Pix
{
    type Output = Pix;

    fn mul(self, rhs: f32) -> Self::Output
    {
        self.map(|x| x*rhs)
    }
}

impl Mul<f32> for &Pix
{
    type Output = Pix;

    fn mul(self, rhs: f32) -> Self::Output
    {
        self.map(|x| x*rhs)
    }
}

impl Mul for Pix
{
    type Output = Pix;

    fn mul(self, rhs: Pix) -> Self::Output
    {
        self.map2(&rhs, |x, y| x*y)
    }
}

impl Mul for &Pix
{
    type Output = Pix;

    fn mul(self, rhs: &Pix) -> Self::Output
    {
        self.map2(rhs, |x, y| x*y)
    }
}


impl Div<f32> for Pix
{
    type Output = Pix;

    fn div(self, rhs: f32) -> Self::Output
    {
        self.map(|x| x/rhs)
    }
}

impl Div<f32> for &Pix
{
    type Output = Pix;

    fn div(self, rhs: f32) -> Self::Output
    {
        self.map(|x| x/rhs)
    }
}




#[derive(Clone)]
struct Image
{
    img: ImageBuffer<Pix, Vec<f32>>,
    w: i32,
    h: i32
}

impl Image
{
    fn new(img: ImageBuffer<Pix, Vec<f32>>) -> Self
    {
        let w = img.width() as i32;
        let h = img.height() as i32;
        Self{img, w, h}
    }

    fn empty(w: u32, h: u32) -> Self
    {
        let img = ImageBuffer::from_pixel(w, h, Pix([0f32; 3]));
        Self{img, w: w as i32, h: h as i32}
    }

    fn open(name: &str) -> Result<Self, Error>
    {
        let img_rgb = image::open(name)?
            .to_rgb();

        let max_val = 255f32;
        let img = ImageBuffer::<Pix, Vec<f32>>
            ::from_fn(img_rgb.width(),
                      img_rgb.height(),
                      |i, j|
                      {
                          let rgb = img_rgb.get_pixel(i, j).channels();
                          Pix([rgb[0] as f32,
                               rgb[1] as f32,
                               rgb[2] as f32])/max_val
                      });
        
        
        Ok(Self::new(img))
    }

    fn save(&self, name: &str) -> Result<(), Error>
    {
        let max_val = 255f32;
        let img = ImageBuffer::<Rgb<u8>, Vec<u8>>
            ::from_fn(self.img.width(),
                      self.img.height(),
                      |i, j|
                      {
                          let pix = self.img.get_pixel(i, j).channels();
                          Rgb([(max_val*pix[0]) as u8,
                               (max_val*pix[1]) as u8,
                               (max_val*pix[2]) as u8])
                      });
        
        Ok(img.save(name)?)
    }
    
    fn pix_coords(&self, i: i32, j: i32) -> (u32, u32)
    {
        ( i.max(0).min(2*self.w -1 - i) as u32,
          j.max(0).min(2*self.h -1 - j) as u32)
    }
    
    fn get_pix(&self, i: i32, j: i32) -> &Pix
    {
        let (i, j) = self.pix_coords(i, j);
        self.img.get_pixel(i, j)
    }

    /// can panic if OOB
    fn set_pix(&mut self, i: i32, j: i32, pix: Pix)
    {
        self.img.put_pixel(i as u32, j as u32, pix);
    }

    fn iter<'a>(&'a self) -> ImageIterator<'a>
    {
        ImageIterator
        {
            img: &self,
            i: 0,
            j: 0
        }
    }
    
}

#[derive(Debug)]
struct Error(String);

macro_rules! impl_error_from {
    ($type:path) => {
        impl From<$type> for Error {
            fn from(error: $type) -> Self
            {
                Self(format!("{}", error))
            }
        }
    };
    (&$type:path) => {
        impl From<&$type> for Error {
            fn from(error: &$type) -> Self
            {
                Self(format!("{}", error))
            }
        }
    };
}

impl_error_from!(image::ImageError);


   

struct Patch
{
    x: i32,
    y: i32,
    w: i32,
    h: i32
}

impl Patch
{
    fn new(x: i32, y: i32, w: i32, h: i32) -> Self
    {
        Self
        {
            x, y, w, h
        }
    }

    fn new_rad(x: i32, y: i32, r: i32) -> Self
    {
        Self::new(x-r, y-r, 2*r+1, 2*r+1)
    }
    
    fn get_pixels<'a>(&self, img: &'a Image) -> Vec<&'a Pix>
    {
        (self.x..(self.x+self.w))
            .map(|i| (self.y..(self.y+self.h)).map(move |j| img.get_pix(i, j)))
            .flatten()
            .collect()
    }

    fn get_pixels_index<'a>(&self, img: &'a Image) -> Vec<((i32, i32), &'a Pix)>
    {
        (self.x..(self.x+self.w))
            .map(|i| (self.y..(self.y+self.h)).map(move |j| ((i, j), img.get_pix(i, j))))
            .flatten()
            .collect()
    }

    
    fn mean(&self, img: &Image) -> Pix
    {
        self.get_pixels(&img)
            .iter()
            .fold(Pix([0f32; 3]), |s, &&p| s+p)/((self.w*self.h) as f32)
    }

    
}

struct ImageIterator<'a>
{
    img: &'a Image,
    i: i32,
    j: i32
}

impl<'a> Iterator for ImageIterator<'a>
{
    type Item = &'a Pix;

    fn next(&mut self) -> Option<Self::Item>
    {
        if self.j == self.img.h
        { // finished case
            None
        }
        else
        {
            let ret = Some(self.img.get_pix(self.i, self.j));

            self.i += 1;

            if self.i == self.img.w
            {
                self.i = 0;
                self.j += 1;
            }
            ret
        }
            
    }
}

fn snr(image_noise: &Image, image_ref: &Image) -> f32
{
    // def snr(x, y):
    //   s =  np.linalg.norm(x - y)
    //   if s == 0:
    //       return "Equal inputs"
    //   return 20 * np.log10(np.linalg.norm(x) /s)

    assert_eq!(image_noise.w, image_ref.w);
    assert_eq!(image_noise.h, image_ref.h); 
/*
    for i in 40..48
    {
        for j in 40..48
        {
            println!("{:?} ", image_noise.get_pix(i, j).0);
        } println!();
    }
  */  
    let mut s = 0f32;
    for (p1, p2) in image_noise.iter().zip(image_ref.iter())
    {
        let diff = p1-p2;
        let square = diff*diff;
        let sq = square.channels();
        s += sq[0] + sq[1] + sq[2];
       // s += diff.0[0]*diff.0[0] + diff.0[1]*diff.0[1] + diff.0[2]*diff.0[2];
    }

    let s = s.sqrt();
    
    let norm_noise = image_noise.iter()
        .fold(0f32, |sum, p|
              {
                  let square = p*p;
                  let sq = square.channels();
                  sum + sq[0]+sq[1]+sq[2]
              }).sqrt();

    println!("s {}  norm {}", s, norm_noise);
    20f32*(norm_noise / s).log10()
    
}


fn naive(img: &Image) -> Image
{
    let radius = 1;
    let n = (img.w*img.h) as usize;
    
    // standart deviation
    let mean = img.iter()
        .fold(0f32, |sum, pix| pix.0[0] + pix.0[1] + pix.0[2])/((3*n) as f32);
    let var = img.iter()
        .fold(0f32, |sum, pix|
              {
                  let r = pix.0[0] - mean;
                  let g = pix.0[1] - mean;
                  let b = pix.0[2] - mean;

                  sum + r*r + g*g + b*b
              })/((3*n) as f32);
    let h = var.sqrt();
    

    
    let h2 = h*h;

    let patches = (0..img.w)
        .map(|i| (0..img.h)
             .map(move |j| Patch::new_rad(i, j, radius)))
        .flatten()
        .collect::<Vec<_>>();

    let means = patches.iter()
        .map(|pat| pat.mean(img))
        .collect::<Vec<_>>();

    // we cannot store all the weights because that would be too heavy

    let gauss = |i: usize, j: usize|
    {
        let diff = means[i] - means[j];
        let square = diff*diff;
        let sq = square.channels();
        (-(sq[0]+sq[1]+sq[2])/h2).exp()
    };

    // C in wikipedia
    let Cs = (0..n)
        .map(|i|
             {
                 println!("Cs: {} / {}  ({}%)", i, n, i*100/n);

                 (0..n).fold(0f32, |sum, j| sum + gauss(i, j))
             })
        .collect::<Vec<_>>();

    let mut denoised = Image::empty(img.w as u32, img.h as u32);

    let pixel = |i: usize| img.get_pix((i%(img.w as usize)) as i32,
                                       (i/(img.w as usize)) as i32);

    
    for y in 0..img.h
    {
        for x in 0..img.w
        {
            let i = (img.w*y + x) as usize;
            println!("writting: {} / {}", i, n);
            let Ci = Cs[i];
            let sum = (0..n).fold(Pix([0f32; 3]), |sum, j|
                                  {
                                      sum + pixel(j)*gauss(i, j)
                                  });
            denoised.set_pix(y, x, sum/Ci);
            
        }
    }
    
    return denoised;
}


// NAIVE APPROACH: snr 3.582176 contre 17.075794 (nul)    !

fn test<F>(name: &str, img_noise: &Image, img_ref: &Image, denoizer: F) -> Result<(), Error>
where
    F: Fn(&Image) -> Image
{
    let img_denoised = denoizer(&img_noise);
    let snr_ref = snr(img_noise, img_ref);
    let snr_den = snr(img_noise, &img_denoised);
    print!("SNRS: {} {}", snr_den, snr_ref);
    img_denoised.save(&format!("denoised/{}.png", name))?;
    Ok(())
}


fn main() -> Result<(), Error>
{
    //let img_bruit = Image::open("images/ricardo_power_bruit.png")?;
    //let img_ref = Image::open("images/ricardo_power.png")?;
    let img_bruit = Image::open("images/ricardo_bruit_sample-1.png")?;
    let img_ref = Image::open("images/ricardo_sample-1.png")?;

    let img = IImage::new("images/ricardo_sample-1.png")?;
    img.save("lol.png")?;
    println!("snr: {}", snr(&img_bruit, &img_ref));
    //test("naive_0", &img_bruit, &img_ref, naive)?;
    
    Ok(())
}
