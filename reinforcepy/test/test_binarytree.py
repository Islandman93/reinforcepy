import pytest

from reinforcepy.handlers.experience_replay.prioritized import Node, BinaryTree


@pytest.fixture(scope='module')
def node():
    root = Node(0, 0)
    return root


def test_insert(node: Node):
    assert node.right is None and node.left is None, "no inserts can be made before this test"

    # right insert
    node.insert(1, 1)
    assert isinstance(node.right, Node)
    assert node.right.value == 1

    # left insert
    node.insert(-1, -1)
    assert isinstance(node.left, Node)
    assert node.left.value == -1


def test_depth(node: Node):
    assert node.depth() == 2


def test_size(node: Node):
    assert node.get_size() == 3


def test_yx_vals(node: Node):
    yx_list = node.get_yx_vals([], 2, 2)
    assert yx_list[0] == [0, 1, 0]
    assert yx_list[1] == [1, 2, 1]
    assert yx_list[2] == [1, 0, -1]


def test_pop_max(node: Node):
    import copy
    test_node = copy.deepcopy(node)
    node_left = test_node.left

    node_val, _, _, _ = test_node.pop_max()
    assert node_val == 1

    node_val, _, hanging_left, terminal = test_node.pop_max()
    assert terminal == 1
    assert hanging_left == node_left
    assert node_val == 0


def test_pop_min(node: Node):
    import copy
    test_node = copy.deepcopy(node)
    node_right = test_node.right

    node_val, _, _, _ = test_node.pop_min()
    assert node_val == -1

    node_val, _, hanging_right, terminal = test_node.pop_min()
    assert terminal == 1
    assert hanging_right == node_right
    assert node_val == 0


def test_update_extra_vals(node: Node):
    def compare(ind):
        return ind > 0

    def update(e_val):
        return e_val + 1
    node.update_extra_vals(compare, update)

    assert node.right.extra_vals == 2
    assert node.extra_vals == 0
    assert node.left.extra_vals == -1


@pytest.fixture(scope='module')
def btree():
    binary_tree = BinaryTree()
    binary_tree.insert(0)
    binary_tree.insert(1)
    binary_tree.insert(-1)
    return binary_tree


def test_binary_tree_pop_max(btree: BinaryTree):
    import copy
    test_tree = copy.deepcopy(btree)
    test_tree.pop_max()
    test_tree.pop_max()
    # binary tree handles hanging left implicitly so we need to test it works
    assert test_tree.root.value == -1


def test_binary_tree_pop_min(btree: BinaryTree):
    import copy
    test_tree = copy.deepcopy(btree)
    test_tree.pop_min()
    test_tree.pop_min()
    # binary tree handles hanging right implicitly so we need to test it works
    assert test_tree.root.value == 1


