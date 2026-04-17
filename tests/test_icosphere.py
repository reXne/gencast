from gencast_repro.geometry.icosphere import create_icosphere


def test_icosphere_shapes():
    vertices, faces = create_icosphere(1)
    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3
    assert vertices.shape[0] > 12

