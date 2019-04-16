import tensorflow as tf
import os


def signature(function_dict):
    signature_dict = {}
    for k, v in function_dict.items():
        inputs = {k: tf.saved_model.utils.build_tensor_info(v) for k, v in v['inputs'].items()}
        outputs = {k: tf.saved_model.utils.build_tensor_info(v) for k, v in v['outputs'].items()}
        signature_dict[k] = tf.saved_model.build_signature_def(inputs=inputs,
                                                               outputs=outputs,
                                                               method_name=v['method_name'])
        return signature_dict


output_dir = 'output_model'
for i in range(100000, 999999999):
    cur = os.path.join(output_dir, str(i))
    if not tf.gfile.Exists(cur):
        output_dir = cur
        break
method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME
print('outputdir', output_dir)
with tf.Graph().as_default(), tf.Session() as sess:
    counter = tf.Variable(0.0, dtype=tf.float32, name="counter")
    with tf.name_scope("incr_counter_op", values=[counter]):
        incr_counter = counter.assign_add(1.0)
    delta = tf.placeholder(dtype=tf.float32, name="delta")
    with tf.name_scope("incr_counter_by_op", values=[counter, delta]):
        incr_counter_by = counter.assign(delta)
    with tf.name_scope("reset_counter_op", values=[counter]):
        reset_counter = counter.assign(0.0)
    nothing = tf.placeholder(dtype=tf.int32, shape=(None,))
    sess.run(tf.global_variables_initializer())
    signature_def_map = signature({
        "incr_counter": {"inputs": {"nothing": nothing},
                         "outputs": {"output": incr_counter},
                         "method_name": method_name},
        "incr_counter_by": {"inputs": {'delta': delta, },
                            "outputs": {'output': incr_counter_by},
                            "method_name": method_name},
        "reset_counter": {"inputs": {"nothing": nothing},
                          "outputs": {"output": reset_counter},
                          "method_name": method_name}
    })
    builder = tf.saved_model.builder.SavedModelBuilder(output_dir)
    builder.add_meta_graph_and_variables(sess, tags=[tf.saved_model.tag_constants.SERVING],
                                         signature_def_map=signature_def_map)
    builder.save()
    print("over")