var_refs:  
[
 <tf.Tensor 'feature_extraction/w_Conv1:0' shape=(8, 8, 4, 16) dtype=float32_ref>,
 <tf.Tensor 'feature_extraction/b_Conv1:0' shape=(16,) dtype=float32_ref>,
 <tf.Tensor 'feature_extraction/w_Conv2:0' shape=(4, 4, 16, 32) dtype=float32_ref>,
 <tf.Tensor 'feature_extraction/b_Conv2:0' shape=(32,) dtype=float32_ref>,
 <tf.Tensor 'feature_extraction/w_fc1:0' shape=(2592, 256) dtype=float32_ref>,
 <tf.Tensor 'feature_extraction/b_fc1:0' shape=(256,) dtype=float32_ref>,
 <tf.Tensor 'imagination/model_function/w_dec:0' shape=(2048, 256) dtype=float32_ref>,
 <tf.Tensor 'imagination/model_function/b_dec:0' shape=(256,) dtype=float32_ref>,
 <tf.Tensor 'imagination/model_function/w_enc:0' shape=(256, 2048) dtype=float32_ref>,
 <tf.Tensor 'imagination/model_function/w_a:0' shape=(3, 2048) dtype=float32_ref>,
 <tf.Tensor 'imagination/model_function/w_r:0' shape=(2048, 1) dtype=float32_ref>,
 <tf.Tensor 'imagination/model_function/b_r:0' shape=(1,) dtype=float32_ref>,
 <tf.Tensor 'imagination/value_function/w_v:0' shape=(256, 1) dtype=float32_ref>,
 <tf.Tensor 'imagination/value_function/b_v:0' shape=(1,) dtype=float32_ref>,
 <tf.Tensor 'policy_function/w_pi:0' shape=(256, 3) dtype=float32_ref>,
 <tf.Tensor 'policy_function/b_pi:0' shape=(3,) dtype=float32_ref>]

gradients:  [
 <tf.Tensor 'gradients/feature_extraction/Conv2D_grad/Conv2DBackpropFilter:0' shape=(8, 8, 4, 16) dtype=float32>,
 <tf.Tensor 'gradients/feature_extraction/add_grad/Reshape_1:0' shape=(16,) dtype=float32>,
 <tf.Tensor 'gradients/feature_extraction/Conv2D_1_grad/Conv2DBackpropFilter:0' shape=(4, 4, 16, 32) dtype=float32>,
 <tf.Tensor 'gradients/feature_extraction/add_1_grad/Reshape_1:0' shape=(32,) dtype=float32>,
 <tf.Tensor 'gradients/feature_extraction/MatMul_grad/MatMul_1:0' shape=(2592, 256) dtype=float32>,
 <tf.Tensor 'gradients/feature_extraction/add_2_grad/Reshape_1:0' shape=(256,) dtype=float32>,
 <tf.Tensor 'gradients/AddN_10:0' shape=(2048, 256) dtype=float32>,
 <tf.Tensor 'gradients/AddN_8:0' shape=(256,) dtype=float32>,
 <tf.Tensor 'gradients/AddN_11:0' shape=(256, 2048) dtype=float32>,
 <tf.Tensor 'gradients/AddN_12:0' shape=(3, 2048) dtype=float32>,
 <tf.Tensor 'gradients/AddN_4:0' shape=(2048, 1) dtype=float32>,
 <tf.Tensor 'gradients/AddN_2:0' shape=(1,) dtype=float32>,
 <tf.Tensor 'gradients/imagination/discounted_return_4/MatMul_grad/MatMul_1:0' shape=(256, 1) dtype=float32>,
 <tf.Tensor 'gradients/imagination/discounted_return_4/add_1_grad/Reshape_1:0' shape=(1,) dtype=float32>,
 <tf.Tensor 'gradients/policy_function/MatMul_5_grad/MatMul_1:0' shape=(256, 3) dtype=float32>,
 <tf.Tensor 'gradients/policy_function/add_5_grad/Reshape_1:0' shape=(3,) dtype=float32>
 ] 
