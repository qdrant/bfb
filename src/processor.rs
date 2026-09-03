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

    /// Number of requests needed for `total_items`. Processors with variable
    /// request sizes can override this to expose a precomputed schedule.
    fn request_count(&self, total_items: usize) -> usize {
        total_items.div_ceil(self.get_batch_size())
    }

    /// Number of items represented by one request (for progress accounting).
    fn request_size(&self, _req_id: usize) -> usize {
        self.get_batch_size()
    }
}
