"""Script to compare two source spaces and check for equality."""

import pickle
from pathlib import Path

from numpy.testing import assert_allclose, assert_array_equal

reference_src_path = Path(__file__).parent / "reference_src.pkl"
new_src_path = Path(__file__).parent / "new_src.pkl"


def assert_source_spaces_equal() -> None:
    """Run the test."""
    with open(reference_src_path, "rb") as f:
        reference_src = pickle.load(f)

    with open(new_src_path, "rb") as f:
        new_src = pickle.load(f)

    # Check the cerebral cortex (index 0) and cerebellum (index 1)
    for i, src_name in enumerate(["cortex", "cerebellum"]):
        print(f"Checking equality of source space for {src_name}...")
        source_space_new = new_src[i]
        source_space_reference = reference_src[i]

        # Check dictionary structure
        assert set(source_space_new.keys()) == set(source_space_reference.keys()), (
            f"Dictionary keys changed for {src_name}!"
        )

        # Check Floats (Coordinates and Normals) with a tolerance
        assert_allclose(
            source_space_new["rr"],
            source_space_reference["rr"],
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Vertices drifted beyond acceptable tolerance for {src_name}!",
        )
        assert_allclose(
            source_space_new["nn"],
            source_space_reference["nn"],
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Normals drifted beyond acceptable tolerance for {src_name}!",
        )

        # C. Check Integers (Triangles and Masks) for exact equality
        assert_array_equal(
            source_space_new["tris"],
            source_space_reference["tris"],
            err_msg=f"Triangle mapping changed for {src_name}!",
        )
        assert_array_equal(
            source_space_new["inuse"],
            source_space_reference["inuse"],
            err_msg=f"In-use mask changed for {src_name}!",
        )


if __name__ == "__main__":
    assert_source_spaces_equal()
    print("Source spaces are equal within the specified tolerances.")
