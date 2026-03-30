//! horus-tf CLI — Coordinate frame inspection and debugging
//!
//! Standalone binary that reads TF data from HORUS shared memory.
//! Installed as a plugin: `horus install horus-tf` registers `frame` and `tf` commands.
//!
//! Usage:
//!   horus-tf frame list          - List all frames
//!   horus-tf frame echo A B      - Echo transform from A to B
//!   horus-tf frame tree          - Show frame tree structure
//!   horus-tf tf A B              - Alias for frame echo

use clap::{Parser, Subcommand};
use horus_core::communication::Topic;
use horus_core::error::HorusResult;
use horus_tf::{TFMessage, Transform, TransformFrame};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const TF_TOPIC: &str = "tf";
const TF_STATIC_TOPIC: &str = "tf_static";

#[derive(Parser)]
#[command(name = "horus-tf", about = "Coordinate frame transforms for HORUS")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Frame inspection commands
    Frame {
        #[command(subcommand)]
        action: FrameAction,
    },
    /// Echo transform between two frames (alias for frame echo)
    Tf {
        /// Source frame
        source: String,
        /// Target frame
        target: String,
        /// Refresh rate in Hz
        #[arg(short, long, default_value = "10")]
        rate: f64,
    },
}

#[derive(Subcommand)]
enum FrameAction {
    /// List all registered frames
    List,
    /// Show frame tree structure
    Tree,
    /// Continuously echo transform between two frames
    Echo {
        source: String,
        target: String,
        #[arg(short, long, default_value = "10")]
        rate: f64,
    },
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Frame { action } => match action {
            FrameAction::List => run_list(),
            FrameAction::Tree => run_tree(),
            FrameAction::Echo { source, target, rate } => run_echo(&source, &target, rate),
        },
        Commands::Tf { source, target, rate } => run_echo(&source, &target, rate),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

/// Collect transforms from SHM topics for a duration, returning a populated TransformFrame
fn collect_frames(collect_duration: Duration) -> HorusResult<TransformFrame> {
    let tf = TransformFrame::new();
    let tf_topic: Topic<TFMessage> = Topic::new(TF_TOPIC)?;
    let tf_static_topic: Topic<TFMessage> = Topic::new(TF_STATIC_TOPIC)?;

    let start = Instant::now();
    while start.elapsed() < collect_duration {
        if let Some(msg) = tf_topic.recv() {
            for ts in msg.iter() {
                let parent = ts.parent_frame_id();
                let child = ts.child_frame_id();
                if !parent.is_empty() && !child.is_empty() {
                    if !tf.has_frame(&parent) {
                        let _ = tf.register_frame(&parent, None);
                    }
                    if !tf.has_frame(&child) {
                        let _ = tf.register_frame(&child, Some(&parent));
                    }
                    let _ = tf.update_transform(&child, &ts.transform, ts.timestamp_ns);
                }
            }
        }
        if let Some(msg) = tf_static_topic.recv() {
            for ts in msg.iter() {
                let parent = ts.parent_frame_id();
                let child = ts.child_frame_id();
                if !parent.is_empty() && !child.is_empty() {
                    if !tf.has_frame(&parent) {
                        let _ = tf.register_frame(&parent, None);
                    }
                    if !tf.has_frame(&child) {
                        let _ = tf.register_static_frame(&child, Some(&parent), &ts.transform);
                    }
                }
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    Ok(tf)
}

fn run_list() -> HorusResult<()> {
    println!("Collecting frames (500ms)...");
    let tf = collect_frames(Duration::from_millis(500))?;
    let frames = tf.all_frames();

    if frames.is_empty() {
        println!("No frames found. Is a HORUS session running with TF publishers?");
        return Ok(());
    }

    println!("Frames ({}):", frames.len());
    for name in &frames {
        println!("  {}", name);
    }
    Ok(())
}

fn run_tree() -> HorusResult<()> {
    println!("Collecting frames (500ms)...");
    let tf = collect_frames(Duration::from_millis(500))?;
    let frames = tf.all_frames();

    if frames.is_empty() {
        println!("No frames found.");
        return Ok(());
    }

    println!("{}", tf.format_tree());
    Ok(())
}

fn run_echo(source: &str, target: &str, rate: f64) -> HorusResult<()> {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::handle(move || r.store(false, Ordering::Relaxed)).ok();

    let tf = TransformFrame::new();
    let tf_topic: Topic<TFMessage> = Topic::new(TF_TOPIC)?;
    let tf_static_topic: Topic<TFMessage> = Topic::new(TF_STATIC_TOPIC)?;

    let period = Duration::from_secs_f64(1.0 / rate);

    println!("Echoing transform {} → {} at {:.0} Hz (Ctrl+C to stop)", source, target, rate);

    while running.load(Ordering::Relaxed) {
        // Drain topics
        while let Some(msg) = tf_topic.recv() {
            for ts in msg.iter() {
                let parent = ts.parent_frame_id();
                let child = ts.child_frame_id();
                if !parent.is_empty() && !child.is_empty() {
                    if !tf.has_frame(&parent) {
                        let _ = tf.register_frame(&parent, None);
                    }
                    if !tf.has_frame(&child) {
                        let _ = tf.register_frame(&child, Some(&parent));
                    }
                    let _ = tf.update_transform(&child, &ts.transform, ts.timestamp_ns);
                }
            }
        }
        while let Some(msg) = tf_static_topic.recv() {
            for ts in msg.iter() {
                let parent = ts.parent_frame_id();
                let child = ts.child_frame_id();
                if !parent.is_empty() && !child.is_empty() {
                    if !tf.has_frame(&parent) {
                        let _ = tf.register_frame(&parent, None);
                    }
                    if !tf.has_frame(&child) {
                        let _ = tf.register_static_frame(&child, Some(&parent), &ts.transform);
                    }
                }
            }
        }

        // Query transform
        match tf.tf(source, target) {
            Ok(t) => {
                let euler = t.to_euler();
                println!(
                    "Translation: [{:.4}, {:.4}, {:.4}]  Rotation (rpy): [{:.4}, {:.4}, {:.4}]",
                    t.translation[0], t.translation[1], t.translation[2],
                    euler[0], euler[1], euler[2],
                );
            }
            Err(_) => {
                println!("Waiting for transform {} → {}...", source, target);
            }
        }

        std::thread::sleep(period);
    }

    println!("\nStopped.");
    Ok(())
}

/// Minimal ctrlc handler (avoids pulling in full ctrlc crate)
mod ctrlc {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    static HANDLER_SET: AtomicBool = AtomicBool::new(false);
    static mut CALLBACK: Option<Box<dyn Fn() + Send>> = None;

    pub fn handle<F: Fn() + Send + 'static>(f: F) -> Result<(), ()> {
        if HANDLER_SET.swap(true, Ordering::SeqCst) {
            return Err(());
        }
        unsafe {
            CALLBACK = Some(Box::new(f));
            libc::signal(libc::SIGINT, handler as libc::sighandler_t);
        }
        Ok(())
    }

    extern "C" fn handler(_: libc::c_int) {
        unsafe {
            if let Some(ref cb) = CALLBACK {
                cb();
            }
        }
    }
}
