import pytest

from rimeX.records import make_models_equiprobable

@pytest.mark.parametrize("records,expected",
                         [
                             (
                                 [
                                     {"model": "A", "variable": "tas", "region": "EU", "warming_level": 1.5},
                                     {"model": "A", "variable": "tas", "region": "EU", "warming_level": 1.5},
                                 ],
                                 [0.5,0.5]
                              ),
                             (
                                 [
                                     {"model": "A", "variable": "tas", "region": "EU", "warming_level": 1.5},
                                     {"model": "A", "variable": "tas", "region": "EU", "warming_level": 1.5},
                                     {"model": "B", "variable": "tas", "region": "EU", "warming_level": 1.5},
                                 ],
                                 [0.5,0.5,1.0]
                              ),
                             (
                                 [
                                     {"model": "A", "variable": "tas", "region": "EU", "warming_level": 1.5, "weight":3},
                                     {"model": "A", "variable": "tas", "region": "EU", "warming_level": 1.5, "weight":3},
                                     {"model": "A", "variable": "tas", "region": "USA", "warming_level": 1.5, "weight":3},
                                 ],
                                 [0.5,0.5,1.0]
                              ),
                             ],
        )
def test_make_models_equiprobable(records,expected):

    make_models_equiprobable(records)

    result = [r["weight"] for r in records]

    assert result == expected
    assert sum(r["weight"] for r in records) == sum(expected)

