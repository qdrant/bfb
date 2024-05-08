use crate::args::Args;
use crate::common::Timing;
use indicatif::ProgressBar;

pub trait Processor {
    async fn make_request(
        &self,
        _req_id: usize,
        args: &Args,
        progress_bar: &ProgressBar,
    ) -> Result<(), anyhow::Error>;

    fn start_timestamp_millis(&self) -> f64;

    fn server_timings(&self) -> Vec<Timing>;

    fn rps(&self) -> Vec<Timing>;

    fn full_timings(&self) -> Vec<Timing>;
}
