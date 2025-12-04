use std::{fmt::Write, path::PathBuf};

use anyhow::{Context, Error as E, Result, bail};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::debertav2::Config as DebertaV2Config;
use clap::Parser;
use console::Style;
use either::Either;
use hf_hub::{Repo, RepoType, api::sync::Api};
use similar::{ChangeTag, TextDiff};
use tokenizers::{Encoding, PaddingParams, Tokenizer};

use candle_shims::utils::device::get_device;
use provence_rs::ProvenceModel;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The model id to use from HuggingFace
    #[arg(
        long,
        default_value = "naver/provence-reranker-debertav3-v1",
        group = "model_source",
        conflicts_with = "model_path"
    )]
    model_id: String,

    /// Local model path
    #[arg(long, group = "model_source", conflicts_with = "model_id")]
    model_path: Option<PathBuf>,

    /// Revision of the model to use (default: "main")
    #[arg(long, default_value = "main")]
    revision: String,

    /// Question
    #[arg(
        short,
        long,
        default_value = "What goes on the bottom of Shepherd's pie?"
    )]
    question: String,

    /// Context (either repeat the flag or provide a comma-separated list)
    #[arg(
        short,
        long,
        num_args = 1..,
        default_values = &[
            "A cottage pie is a type of meat pie made with minced or ground beef and topped with mashed potato. The dish is also known as shepherd's pie when made with lamb. Shepherd's pie is traditionally made with lamb, while cottage pie uses beef. Both are topped with mashed potatoes and baked until golden. In Quebec, a variation on the cottage pie is called \"Paˆte ́ chinois\". It is made with ground beef on the bottom layer, canned corn in the middle, and mashed potato on top.",
            "Shepherd's pie. History. In early cookery books, the dish was a means of using leftover roasted meat of any kind, and the pie dish was lined on the sides and bottom with mashed potato, as well as having a mashed potato crust on top.",
            "Variations and similar dishes. Other potato-topped pies include: The modern \"Cumberland pie\" is a version with either beef or lamb and a layer of bread- crumbs and cheese on top. In medieval times, and modern-day Cumbria, the pastry crust had a filling of meat with fruits and spices..",
            "The \"shepherdess pie\" is a vegetarian version made without meat, or a vegan version made without meat and dairy.. In the Netherlands, a very similar dish called \"philosopher's stew\" () often adds ingredients like beans, apples, prunes, or apple sauce.. In Brazil, a dish called in refers to the fact that a manioc puree hides a layer of sun-dried meat.",
            "Traditional shepherd's pie often includes diced onions, carrots, and peas mixed into the minced lamb on the bottom layer, enhancing flavor and texture before the mashed potato topping is added. Modern variations of Shepherd's pie may use sweet potato instead of mashed potato for the topping, giving a slightly sweeter taste and different nutritional profile.",
            "Vegetarian or vegan shepherd's pies replace the meat with lentils, mushrooms, or textured vegetable protein, while keeping the layered structure of a bottom filling, vegetables, and mashed potato on top. Shepherd's pie can also include a layer of gravy or sauce on the bottom to keep the filling moist, which helps prevent the mashed potato from drying out during baking.",
            "In the UK, Shepherd's pie is considered comfort food and is often served with a side of peas or a simple green salad, especially during colder months. Cultural variations: In Ireland, shepherd's pie may incorporate Irish stout into the meat mixture for deeper flavor; in Canada, 'Pâté chinois' is a common dish in Quebec, with corn as the middle layer."
        ]
    )]
    contexts: Vec<String>,

    /// Threshold
    #[arg(short, long, default_value = "0.5")]
    threshold: f32,

    /// Always select first sentence
    #[arg(long, default_value_t = true)]
    always_select_first: bool,
}

impl Args {
    fn build_model_and_tokenizer(&self) -> Result<(ProvenceModel, DebertaV2Config, Tokenizer)> {
        let device = get_device(self.cpu, false)?;

        // Get files from either the HuggingFace API, or from a specified local directory.
        let (config_filename, tokenizer_filename, weights_filename) =
            get_model_files(&self.model_path, &self.model_id, &self.revision)?;

        let config = std::fs::read_to_string(config_filename)?;
        let config: DebertaV2Config = serde_json::from_str(&config)?;

        let id2label = if let Some(id2label) = &config.id2label {
            id2label.clone()
        } else {
            bail!("Id2Label not found in the model configuration nor specified as a parameter")
        };

        let mut tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenizer error: {e}")))?;

        tokenizer.with_padding(Some(PaddingParams::default()));

        #[allow(unsafe_code)]
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[weights_filename],
                candle_transformers::models::debertav2::DTYPE,
                &device,
            )?
        };

        let vb = vb.set_prefix("deberta");

        Ok((
            ProvenceModel::load(vb, &config, Some(id2label.clone()))?,
            config,
            tokenizer,
        ))
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let model_load_time = std::time::Instant::now();
    let (model, _model_config, tokenizer) = args.build_model_and_tokenizer()?;

    println!(
        "Loaded model and tokenizers in {:?}",
        model_load_time.elapsed()
    );

    let tokenize_time = std::time::Instant::now();

    println!(
        "Tokenized and loaded inputs in {:?}",
        tokenize_time.elapsed()
    );

    let question = &args.question;
    let first_context = args.contexts.first().context("context can't be empty")?;

    println!("Running forward pass only on question and first context");

    let input_text = ProvenceModel::apply_template(question, first_context);

    let encoding = tokenizer
        .encode(input_text, true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

    let input_ids = Tensor::new(encoding.get_ids(), &model.device)?.unsqueeze(0)?;
    let attention_mask = Tensor::new(encoding.get_attention_mask(), &model.device)?.unsqueeze(0)?;

    let output = model.forward(&input_ids, Some(attention_mask.clone()))?;

    println!("Forward pass output");
    dbg!(&output);

    println!("Running process helper function");

    dbg!(&question);

    let question = vec![question.to_string(), "What is a cottage pie?".to_string()];

    let start = std::time::Instant::now();

    let result = model.process(
        &tokenizer,
        Either::Left(question),
        Either::Left(vec![args.contexts.clone(), args.contexts.clone()]),
        None,
        Some(args.threshold),
        Some(args.always_select_first),
        None,
        None,
        None,
        None,
    )?;

    let duration = start.elapsed();

    println!("Simple output");

    dbg!(&result);

    for inner_pruned_contexts in result.pruned_context.iter() {
        for (pruned_context_index, pruned_context) in inner_pruned_contexts.iter().enumerate() {
            let original_context = args
                .contexts
                .get(pruned_context_index)
                .context("Couldn't get original context")?;

            let diff = get_inline_diff(original_context, pruned_context)?;
            println!("\n{}", diff);
        }
    }

    println!("\nTime elapsed: {:?}", duration);

    Ok(())
}

fn get_model_files(
    model_path: &Option<PathBuf>,
    model_id: &str,
    revision: &str,
) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let config_filename = "config.json";
    let tokenizer_filename = "tokenizer.json";
    let weights_filename = "model.safetensors";

    let config;
    let tokenizer;
    let weights;

    match model_path {
        Some(base_path) => {
            if !base_path.is_dir() {
                bail!("Model path {} is not a directory.", base_path.display())
            }

            config = base_path.join(config_filename);
            tokenizer = base_path.join(tokenizer_filename);
            weights = base_path.join(weights_filename);
        }
        None => {
            let repo =
                Repo::with_revision(model_id.to_owned(), RepoType::Model, revision.to_owned());

            let api = Api::new()?;
            let api = api.repo(repo);

            config = api.get(config_filename)?;
            tokenizer = api.get(tokenizer_filename)?;
            weights = api.get(weights_filename)?;
        }
    }

    Ok((config, tokenizer, weights))
}

fn get_inline_diff(old: &str, new: &str) -> Result<String> {
    let mut output = String::new();

    let diff = TextDiff::from_words(old, new);

    for change in diff.iter_all_changes() {
        let (change_value, change_style) = match change.tag() {
            ChangeTag::Delete => (change.value(), Style::new().red().strikethrough()),
            ChangeTag::Insert => (change.value(), Style::new().green().bold()),
            ChangeTag::Equal => (change.value(), Style::new().dim()),
        };

        write!(output, "{}", change_style.apply_to(change_value))?;
    }

    Ok(output)
}

// From xml-roberta
#[derive(Debug)]
pub enum TokenizeInput<'a> {
    Single(&'a [String]),
    Pairs(&'a [(String, String)]),
}

pub fn tokenize_batch(
    tokenizer: &Tokenizer,
    input: TokenizeInput,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let tokens = get_tokens(tokenizer, input)?;

    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Tensor::new(tokens.as_slice(), device)
        })
        .collect::<candle_core::Result<Vec<_>>>()?;

    Ok(Tensor::stack(&token_ids, 0)?)
}

pub fn get_attention_mask(
    tokenizer: &Tokenizer,
    input: TokenizeInput,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let tokens = get_tokens(tokenizer, input)?;

    let attention_mask = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_attention_mask().to_vec();
            Tensor::new(tokens.as_slice(), device)
        })
        .collect::<candle_core::Result<Vec<_>>>()?;

    Ok(Tensor::stack(&attention_mask, 0)?)
}

fn get_tokens(tokenizer: &Tokenizer, input: TokenizeInput) -> anyhow::Result<Vec<Encoding>> {
    let tokens = match input {
        TokenizeInput::Single(text_batch) => tokenizer
            .encode_batch(text_batch.to_vec(), true)
            .map_err(E::msg)?,
        TokenizeInput::Pairs(pairs) => tokenizer
            .encode_batch(pairs.to_vec(), true)
            .map_err(E::msg)?,
    };

    Ok(tokens)
}
