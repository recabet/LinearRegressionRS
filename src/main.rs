extern crate nalgebra as na;
use na::{DMatrix, DVector};
use std::error::Error;
use std::fmt;

#[derive(Debug)]
enum LinearRegressionError {
    DimensionMismatch,
    Other(String),
}

impl fmt::Display for LinearRegressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Error for LinearRegressionError {}

enum Features {
    Single(DVector<f32>),
    Multiple(DMatrix<f32>),
}

struct LinearRegression {
    alpha: f32,
    iterations: u32,
    w: DVector<f32>,
    b: f32,
}

impl LinearRegression {
    fn new(alpha: f32, iterations: u32, num_features: usize) -> Self {
        Self {
            alpha,
            iterations,
            w: DVector::zeros(num_features),
            b: 0.0,
        }
    }

    fn cost(y: &DVector<f32>, y_pred: &DVector<f32>) -> f32 {
        let m: f32 = y.len() as f32;
        let err: DVector<f32> = y.clone() - y_pred.clone();
        (1.0 / (2.0 * m)) * err.dot(&err)
    }

    fn fit(&mut self, x_train: Features, y_train: DVector<f32>, show_cost: bool, __debug:bool) -> Result<(), LinearRegressionError> {
        let m: f32 = y_train.len() as f32;
        match x_train {
            Features::Single(x_train_vector) => {
                if x_train_vector.len() != y_train.len() {
                    return Err(LinearRegressionError::DimensionMismatch);
                }
                for i in 0..self.iterations {
                    let y_pred: DVector<f32> = &x_train_vector * self.w[0] + DVector::from_element(x_train_vector.len(), self.b);
                    let error: DVector<f32> = y_train.clone() - y_pred.clone();
                    if show_cost {
                        let cost: f32 = LinearRegression::cost(&y_train, &y_pred);
                        println!("Cost for iteration {:?}: {:?}", i + 1, cost);
                    }
                    let dw: f32 = (1.0 / m) * x_train_vector.dot(&error);
                    let db: f32 = (1.0 / m) * error.sum();
                    self.w[0] += self.alpha * dw;
                    self.b += self.alpha * db;
                    if __debug{
                        println!("dw:{:?}, db:{:?} for iteration {:?}", dw, db,i+1);
                        println!("w:{:?}, b:{:?} for iteration {:?}", self.w, self.b,i+1);
                    }
                }
            }
            Features::Multiple(x_train_matrix) => {
                if x_train_matrix.nrows() != y_train.len() {
                    return Err(LinearRegressionError::DimensionMismatch);
                }
                for i in 0..self.iterations {
                    let y_pred: DVector<f32> = x_train_matrix.clone() * &self.w + DVector::from_element(x_train_matrix.nrows(), self.b);
                    let error: DVector<f32> = y_train.clone() - y_pred.clone();
                    if show_cost {
                        let cost: f32 = LinearRegression::cost(&y_train, &y_pred);
                        println!("Cost for iteration {:?}: {:?}", i + 1, cost);
                    }
                    let dw: DVector<f32> = (1.0 / m) * x_train_matrix.transpose() * &error;
                    let db: f32 = (1.0 / m) * error.sum();
                    self.w += self.alpha * dw;
                    self.b += self.alpha * db;
                    if __debug{
                        println!("dw:{:?}, db:{:?} for iteration {:?}", dw, db,i+1);
                        println!("w:{:?}, b:{:?} for iteration {:?}", self.w, self.b,i+1);
                    }
                }
            }
        }
        Ok(())
    }

    fn predict(&self, x: Features) -> Result<DVector<f32>, LinearRegressionError> {
        match x {
            Features::Single(x_vector) => {
                Ok(&x_vector * self.w[0] + DVector::from_element(x_vector.len(), self.b))
            }
            Features::Multiple(x_matrix) => {
                if x_matrix.ncols() != self.w.len() {
                    return Err(LinearRegressionError::DimensionMismatch);
                }
                Ok(x_matrix.clone() * &self.w + DVector::from_element(x_matrix.nrows(), self.b))
            }
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let x_train_vector: DVector<f32> = DVector::from_column_slice(&[1.0, 2.0, 3.0]);
    let y_train_vector: DVector<f32> = DVector::from_column_slice(&[2.0, 4.0, 6.0]);

    let mut model = LinearRegression::new(0.01, 1000, 1);
    model.fit(Features::Single(x_train_vector.clone()), y_train_vector.clone(), false,false)?;

    let y_pred_vector = model.predict(Features::Single(x_train_vector.clone()))?;
    println!("Predictions with vector input: {:?}", y_pred_vector);

    let x_train_matrix: DMatrix<f32> = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let y_train_matrix: DVector<f32> = DVector::from_column_slice(&[1.0, 2.0, 3.0]);

    let mut model_multiple = LinearRegression::new(0.01, 1000, x_train_matrix.ncols());
    model_multiple.fit(Features::Multiple(x_train_matrix.clone()), y_train_matrix.clone(), true,false)?;

    let y_pred_matrix = model_multiple.predict(Features::Multiple(x_train_matrix.clone()))?;
    println!("Predictions with matrix input: {:?}", y_pred_matrix);

    Ok(())
}
