ls -1 *.png | while read line; do pre=`echo $line | grep -Po 'F\K(\d\d)'`; post=`echo $line|grep -Po '\-\K(\d+)'`; new_index=`echo \($pre-61\)*100+$post|bc`; new_name=`printf "%05d.png" $new_index`; echo mv $line $new_name; done

~/Development/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=./saved_models/fcn8_weighted/fcn_weighted_model.h5.pb \
--out_graph=out_graph.pb \
--inputs=input_2 \
--outputs=y_/truediv \
--transforms='
add_default_attributes
remove_nodes(op=Identity, op=CheckNumerics)
fold_constants(ignore_errors=true)
fold_batch_norms
fold_old_batch_norms
fuse_resize_and_conv
strip_unused_nodes'
