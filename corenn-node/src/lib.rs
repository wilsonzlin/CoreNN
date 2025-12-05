use libcorenn::{cfg::{Cfg, CompressionMode}, metric::StdMetric, CoreNN};
use neon::{handle::Handle, object::Object, prelude::{Context, FunctionContext, ModuleContext}, result::NeonResult, types::{buffer::TypedArray, Finalize, JsArray, JsBox, JsNumber, JsObject, JsString, JsTypedArray, JsUndefined, Value}};

// Neon requires Finalize.
struct CoreNNWrapper(CoreNN);

impl Finalize for CoreNNWrapper {}

fn compression_mode_from_str(cx: &mut FunctionContext, s: Handle<JsString>) -> NeonResult<CompressionMode> {
  let s = s.value(cx);
  match s.as_str() {
    "pq" => Ok(CompressionMode::PQ),
    "trunc" => Ok(CompressionMode::Trunc),
    _ => cx.throw_type_error("Invalid compression mode"),
  }
}

fn metric_from_str(cx: &mut FunctionContext, s: Handle<JsString>) -> NeonResult<StdMetric> {
  let s = s.value(cx);
  match s.as_str() {
    "l2" => Ok(StdMetric::L2),
    "cosine" => Ok(StdMetric::Cosine),
    _ => cx.throw_type_error("Invalid metric"),
  }
}

fn as_usize(cx: &mut FunctionContext, v: Handle<JsNumber>) -> NeonResult<usize> {
  let v = v.value(cx);
  if !v.is_finite() || v.fract() != 0.0 {
    return cx.throw_type_error("Expected an integer");
  }
  Ok(v as usize)
}

fn cfg_from_js(cx: &mut FunctionContext, cfg_js: &JsObject) -> NeonResult<Cfg> {
  let mut cfg = Cfg::default();
  macro_rules! prop {
      ($name:ident, $type:ty, $parser:expr) => {
        let maybe = prop::<$type, _>(cx, &cfg_js, stringify!($name), $parser)?;
        if let Some(v) = maybe {
          cfg.$name = v;
        }
      };
  }
  prop!(beam_width, JsNumber, |cx, v| as_usize(cx, v));
  prop!(compaction_threshold_deletes, JsNumber, |cx, v| as_usize(cx, v));
  prop!(compression_mode, JsString, |cx, v| compression_mode_from_str(cx, v));
  prop!(compression_threshold, JsNumber, |cx, v| as_usize(cx, v));
  prop!(dim, JsNumber, |cx, v| as_usize(cx, v));
  prop!(max_add_edges, JsNumber, |cx, v| as_usize(cx, v));
  prop!(max_edges, JsNumber, |cx, v| as_usize(cx, v));
  prop!(metric, JsString, |cx, v| metric_from_str(cx, v));
  prop!(pq_sample_size, JsNumber, |cx, v| as_usize(cx, v));
  prop!(pq_subspaces, JsNumber, |cx, v| as_usize(cx, v));
  prop!(query_search_list_cap, JsNumber, |cx, v| as_usize(cx, v));
  prop!(trunc_dims, JsNumber, |cx, v| as_usize(cx, v));
  prop!(update_search_list_cap, JsNumber, |cx, v| as_usize(cx, v));
  Ok(cfg)
}

fn prop<V: Value, R>(cx: &mut FunctionContext, obj: &JsObject, key: &str, parser: impl FnOnce(&mut FunctionContext, Handle<V>) -> NeonResult<R>) -> NeonResult<Option<R>> {
  let Some(prop) = obj.get_opt::<V, _, _>(cx, key)? else {
    return Ok(None);
  };
  parser(cx, prop).map(Some)
}

fn create_db(mut cx: FunctionContext) -> NeonResult<Handle<JsBox<CoreNNWrapper>>> {
  let path = cx.argument::<JsString>(0)?.value(&mut cx);
  let cfg_js = cx.argument::<JsObject>(1)?;
  let cfg = cfg_from_js(&mut cx, &cfg_js)?;
  let db = CoreNN::create(path, cfg);
  Ok(JsBox::new(&mut cx, CoreNNWrapper(db)))
}

fn open_db(mut cx: FunctionContext) -> NeonResult<Handle<JsBox<CoreNNWrapper>>> {
  let path = cx.argument::<JsString>(0)?.value(&mut cx);
  let db = CoreNN::open(path);
  Ok(JsBox::new(&mut cx, CoreNNWrapper(db)))
}

fn new_in_memory(mut cx: FunctionContext) -> NeonResult<Handle<JsBox<CoreNNWrapper>>> {
  let cfg_js = cx.argument::<JsObject>(0)?;
  let cfg = cfg_from_js(&mut cx, &cfg_js)?;
  let db = CoreNN::new_in_memory(cfg);
  Ok(JsBox::new(&mut cx, CoreNNWrapper(db)))
}

fn insert(mut cx: FunctionContext) -> NeonResult<Handle<JsUndefined>> {
  let db = &cx.argument::<JsBox<CoreNNWrapper>>(0)?.0;
  let entries = cx.argument::<JsArray>(1)?;
  for entry in entries.to_vec(&mut cx)? {
    let entry = entry.downcast_or_throw::<JsObject, _>(&mut cx)?;
    let key = entry.get::<JsString, _, _>(&mut cx, "key")?.value(&mut cx);
    let vector = entry.get::<JsObject, _, _>(&mut cx, "vector")?;
    if let Ok(as_f32) = vector.downcast::<JsTypedArray<f32>, _>(&mut cx) {
      let vector = as_f32.as_slice(&mut cx);
      db.insert(&key, vector);
    } else if let Ok(as_f64) = vector.downcast::<JsTypedArray<f64>, _>(&mut cx) {
      let vector = as_f64.as_slice(&mut cx);
      db.insert(&key, vector);
    } else {
      cx.throw_type_error("Expected a Float32Array or Float64Array")?;
    }
  }
  Ok(cx.undefined())
}

fn query(mut cx: FunctionContext) -> NeonResult<Handle<JsArray>> {
  let db = &cx.argument::<JsBox<CoreNNWrapper>>(0)?.0;
  let query = cx.argument::<JsObject>(1)?;
  let k = cx.argument::<JsNumber>(2)?;
  let k = as_usize(&mut cx, k)?;
  let results = if let Ok(as_f32) = query.downcast::<JsTypedArray<f32>, _>(&mut cx) {
    let query = as_f32.as_slice(&mut cx);
    db.query(query, k)
  } else if let Ok(as_f64) = query.downcast::<JsTypedArray<f64>, _>(&mut cx) {
    let query = as_f64.as_slice(&mut cx);
    db.query(query, k)
  } else {
    return cx.throw_type_error("Expected a Float32Array or Float64Array");
  };
  let js_results = cx.empty_array();
  for (i, (key, dist)) in results.into_iter().enumerate() {
    let js_result = cx.empty_object();
    let key = cx.string(&key);
    let dist = cx.number(dist);
    js_result.set(&mut cx, "key", key)?;
    js_result.set(&mut cx, "distance", dist)?;
    js_results.set(&mut cx, i as u32, js_result)?;
  }
  Ok(js_results)
}

#[neon::main]
fn main(mut cx: ModuleContext) -> NeonResult<()> {
  cx.export_function("create_db", create_db)?;
  cx.export_function("open_db", open_db)?;
  cx.export_function("new_in_memory", new_in_memory)?;
  cx.export_function("insert", insert)?;
  cx.export_function("query", query)?;
  Ok(())
}
