"""Convert Perch SavedModel to TFLite with dynamic-range quantisation."""
import sys, traceback
import numpy as np

try:
    import tensorflow as tf
    print(f'TF {tf.__version__}', flush=True)

    WIN = 5 * 32000
    MODEL_DIR = 'models/perch/model'
    OUT_PATH  = 'models/perch/perch_v4_dynamic.tflite'

    print('Converting SavedModel → TFLite (Flex delegate for JAX ops) ...', flush=True)
    conv = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)  # uses 'serving_default'
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    # Perch is JAX-converted and uses Transpose/StridedSlice/EnsureShape
    # which require the SELECT_TF_OPS (Flex) delegate
    conv.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    conv._experimental_lower_tensor_list_ops = False
    tflite_bytes = conv.convert()

    with open(OUT_PATH, 'wb') as f:
        f.write(tflite_bytes)
    print(f'Saved {len(tflite_bytes)/1e6:.1f} MB → {OUT_PATH}', flush=True)

    print('Testing batched invoke ...', flush=True)
    interp   = tf.lite.Interpreter(model_path=OUT_PATH, num_threads=4)
    print('Input details:', interp.get_input_details(), flush=True)
    print('Output details:', [(d['name'], d['shape']) for d in interp.get_output_details()], flush=True)
    inp_idx  = interp.get_input_details()[0]['index']
    interp.resize_tensor_input(inp_idx, [4, WIN])
    interp.allocate_tensors()
    interp.set_tensor(inp_idx, np.zeros((4, WIN), dtype=np.float32))
    interp.invoke()
    od       = interp.get_output_details()
    lk       = max(od, key=lambda d: int(np.prod(d['shape'])))
    result   = interp.get_tensor(lk['index'])
    print(f'Batch test OK: output shape={result.shape}  name={lk["name"]}', flush=True)
    print('DONE', flush=True)

except Exception:
    traceback.print_exc()
    sys.exit(1)
