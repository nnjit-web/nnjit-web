
from tvm import relay
from .init import create_workload, get_conv2d_data_layout
from . import layers as wrapper


def get_net(batch_size, image_shape, num_classes, dtype, batch_norm=False):
    data_shape = (batch_size,) + image_shape
    data = relay.var("data", shape=data_shape, dtype=dtype)

    data_layout, kernel_layout, bias_add_axis = get_conv2d_data_layout()

    conv1 = wrapper.conv2d(
        data=data,
        kernel_size=(11, 11),
        strides=(4, 4),
        padding=(1, 1),
        channels=96,
        name="conv1",
        data_layout=data_layout,
        kernel_layout=kernel_layout
    )
    bias1 = relay.nn.bias_add(
        conv1, relay.var("conv1_bias"), bias_add_axis
    )
    relu1 = relay.nn.relu(data=bias1)
    pool1 = relay.nn.max_pool2d(data=relu1, pool_size=(3, 3), strides=(2, 2), layout=data_layout)
    conv2 = wrapper.conv2d(
        data=pool1,
        kernel_size=(5, 5),
        padding=(2, 2),
        channels=256,
        name="conv2",
        data_layout=data_layout,
        kernel_layout=kernel_layout
    )
    bias2 = relay.nn.bias_add(
        conv2, relay.var("conv2_bias"), bias_add_axis
    )
    relu2 = relay.nn.relu(data=bias2)
    pool2 = relay.nn.max_pool2d(data=relu2, pool_size=(3, 3), strides=(2, 2), layout=data_layout)
    conv3 = wrapper.conv2d(
        data=pool2,
        kernel_size=(3, 3),
        padding=(1, 1),
        channels=384,
        name="conv3",
        data_layout=data_layout,
        kernel_layout=kernel_layout
    )
    bias3 = relay.nn.bias_add(
        conv3, relay.var("conv3_bias"), bias_add_axis
    )
    relu3 = relay.nn.relu(data=bias3)
    conv4 = wrapper.conv2d(
        data=relu3,
        kernel_size=(3, 3),
        padding=(1, 1),
        channels=384,
        name="conv4",
        data_layout=data_layout,
        kernel_layout=kernel_layout
    )
    bias4 = relay.nn.bias_add(
        conv4, relay.var("conv4_bias"), bias_add_axis
    )
    relu4 = relay.nn.relu(data=bias4)
    conv5 = wrapper.conv2d(
        data=relu4,
        kernel_size=(3, 3),
        padding=(1, 1),
        channels=256,
        name="conv5",
        data_layout=data_layout,
        kernel_layout=kernel_layout
    )
    bias5 = relay.nn.bias_add(
        conv5, relay.var("conv5_bias"), bias_add_axis
    )
    relu5 = relay.nn.relu(data=bias5)
    pool5 = relay.nn.max_pool2d(data=relu5, pool_size=(3, 3), strides=(2, 2), layout=data_layout)
    flatten = relay.nn.batch_flatten(data=pool5)
    fc6 = wrapper.dense_add_bias(data=flatten, units=4096, name="fc6")
    relu6 = relay.nn.relu(data=fc6)
    fc7 = wrapper.dense_add_bias(data=relu6, units=4096, name="fc7")
    relu7 = relay.nn.relu(data=fc7)
    fc8 = wrapper.dense_add_bias(data=relu7, units=num_classes, name="fc8")

    classifier = fc8
    symbol = relay.nn.softmax(data=classifier)
    symbol = classifier
    args = relay.analysis.free_vars(symbol)
    return relay.Function(args, symbol)


def get_workload(
    batch_size,
    num_classes=1000,
    image_shape=(3, 224, 224),
    dtype="float32",
    batch_norm=False,
):
    net = get_net(batch_size, image_shape, num_classes, dtype, batch_norm)
    return create_workload(net)
