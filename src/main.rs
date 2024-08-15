extern crate nalgebra as na;
use na::{DMatrix, DVector};
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
        let err: DVector<f32> = y - y_pred;
        (1.0 / (2.0 * m)) * err.dot(&err)
    }

    fn fit(&mut self, x_train: Features, y_train: DVector<f32>) {
        let m: f32 = y_train.len() as f32;
        match x_train {
            Features::Single(x_train_vector) => {
                for _ in 0..self.iterations {
                    let y_pred: DVector<f32> = &x_train_vector * self.w[0] + DVector::from_element(x_train_vector.len(), self.b);
                    let error: DVector<f32> = y_train.clone() - y_pred;
                    let dw: f32 = (1.0 / m) * x_train_vector.dot(&error);
                    let db: f32 = (1.0 / m) * error.sum();
                    self.w[0] += self.alpha * dw;
                    self.b += self.alpha * db;
                }
            }
            Features::Multiple(x_train_matrix) => {
                for _ in 0..self.iterations {
                    let y_pred: DVector<f32> = &x_train_matrix * &self.w + DVector::from_element(x_train_matrix.nrows(), self.b);
                    let error: DVector<f32> = y_train.clone() - y_pred;
                    let dw: DVector<f32> = (1.0 / m) * x_train_matrix.transpose() * &error;
                    let db: f32 = (1.0 / m) * error.sum();
                    self.w += self.alpha * dw;
                    self.b += self.alpha * db;
                }
            }
        }
    }

    fn predict(&self, x: Features) -> DVector<f32> {
        match x {
            Features::Single(x_vector) => {
                &x_vector * self.w[0] + DVector::from_element(x_vector.len(), self.b)
            }
            Features::Multiple(x_matrix) => {
                &x_matrix * &self.w + DVector::from_element(x_matrix.nrows(), self.b)
            }
        }
    }
}

fn main() {
    let x_train_vector: DVector<f32> = DVector::from_column_slice(&[1.0, 2.0, 3.0]);
    let y_train_vector: DVector<f32> = DVector::from_column_slice(&[2.0, 4.0, 6.0]);

    let mut model: LinearRegression = LinearRegression::new(0.01, 1000, 1);
    model.fit(Features::Single(x_train_vector.clone()), y_train_vector.clone());
    let y_pred_vector: DVector<f32> = model.predict(Features::Single(x_train_vector.clone()));
    println!("Predictions with vector input: {:?}", y_pred_vector);

    let x_train_matrix: DMatrix<f32> = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let y_train_matrix: DVector<f32> = DVector::from_column_slice(&[1.0, 2.0, 3.0]);

    let mut model_multiple: LinearRegression = LinearRegression::new(0.01, 1000, x_train_matrix.ncols());
    model_multiple.fit(Features::Multiple(x_train_matrix.clone()), y_train_matrix.clone());
    let y_pred_matrix: DVector<f32> = model_multiple.predict(Features::Multiple(x_train_matrix.clone()));
    println!("Predictions with matrix input: {:?}", y_pred_matrix);
}
