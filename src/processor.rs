use crate::args::Args;
use indicatif::ProgressBar;

/// A single measurement: a value observed `delay_millis` after the run start.
#[derive(Debug, Clone)]
pub struct Timing {
    pub delay_millis: u32,
    pub value: f32,
}

pub trait Processor {
    async fn make_request(
        &self,
        _req_id: usize,
        args: &Args,
        progress_bar: &ProgressBar,
    ) -> Result<(), anyhow::Error>;

    fn start_timestamp_millis(&self) -> f64;

    /// Request timing reported by server.
    fn server_timings(&self) -> Vec<Timing>;

    /// Query per second timing time series.
    fn qps(&self) -> Vec<Timing>;
    /// Request per second timing time series.
    fn rps(&self) -> Vec<Timing>;
    /// Timing length time series.
    fn full_timings(&self) -> Vec<Timing>;

    fn precisions(&self) -> Vec<f32> {
        vec![]
    }

    fn get_batch_size(&self) -> usize;
}
