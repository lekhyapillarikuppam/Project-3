"""Tests for zip layout recognition and safe extraction (SPEC 4.1), using a local zip."""

import zipfile
from types import SimpleNamespace

import pytest

from ffs import remote_zip as rz


def test_scene_of():
    assert rz.scene_of("th_0007/img/IMG_0001.JPG") == "th_0007"
    assert rz.scene_of("Th_0012/IPC/IPC_ENG/list.xlsx") == "th_0012"
    assert rz.scene_of("img/th_0003/IMG_1.jpg") == "th_0003"
    assert rz.scene_of("IPC/IPC_ENG/th_0004_IPC_ENG.xlsx") == "th_0004"
    assert rz.scene_of("description/summary.xlsx") is None


@pytest.mark.parametrize("name,kind", [
    ("th_0001/img/IMG_0001.JPG", "photo"),
    ("th_0001/IPC/IPC_ENG/list_ENG.xlsx", "exhibit_list"),
    ("IPC/th_0001_IPC_ENG.xlsx", "exhibit_list"),
    ("description/scenes_ENG.xlsx", "description"),
    ("description/scenes_FR.xlsx", "blocked"),
    ("th_0001/IPC/IPC_FR/liste.xlsx", "blocked"),
    ("th_0001/pano/p.jpg", "blocked"),
    ("th_0001/scan/cloud.las", "blocked"),
    ("th_0001/img/", "other"),
    ("th_0001/notes.txt", "other"),
])
def test_kind_of(name, kind):
    assert rz.kind_of(name) == kind


def test_wanted_respects_scene_filter():
    assert rz.wanted("th_0001/img/a.jpg", {"th_0001"})
    assert not rz.wanted("th_0002/img/a.jpg", {"th_0001"})
    assert rz.wanted("th_0002/img/a.jpg", None)
    assert rz.wanted("description/scenes_ENG.xlsx", {"th_0001"})   # global, small
    assert not rz.wanted("th_0001/scan/cloud.las", None)


def test_safe_dest_blocks_path_tricks(tmp_path):
    assert rz.safe_dest(tmp_path, "th_0001/img/a.jpg").is_relative_to(tmp_path)
    for bad in ["../evil.txt", "th_0001/../../evil.txt", "C:/evil.txt"]:
        with pytest.raises(ValueError):
            rz.safe_dest(tmp_path, bad)


def make_zip(path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("th_0001/img/a.jpg", b"A" * 100)
        z.writestr("th_0001/img/b.jpg", b"B" * 200)
        z.writestr("th_0001/IPC/IPC_ENG/list_ENG.csv", "1,knife\n")
        z.writestr("th_0001/scan/cloud.las", b"X" * 5000)
        z.writestr("th_0002/img/c.jpg", b"C" * 300)
    return path


def test_extract_wanted_downloads_only_selected_and_skips_existing(tmp_path):
    zpath = make_zip(tmp_path / "t.zip")
    out = tmp_path / "raw"
    with zipfile.ZipFile(zpath) as z:
        infos = z.infolist()
        plan = rz.extract_wanted(z, infos, out, {"th_0001"}, dry_run=True)
        assert plan["planned"] == 3 and plan["planned_bytes"] == 100 + 200 + len("1,knife\n")

        first = rz.extract_wanted(z, infos, out, {"th_0001"})
        assert first["downloaded"] == 3 and first["skipped"] == 0
        assert (out / "th_0001/img/a.jpg").read_bytes() == b"A" * 100
        assert not (out / "th_0001/scan/cloud.las").exists()    # R-A0-1
        assert not (out / "th_0002").exists()

        second = rz.extract_wanted(z, infos, out, {"th_0001"})   # R-A0-3
        assert second["downloaded"] == 0 and second["skipped"] == 3
    assert not list(out.rglob("*.part"))


def test_summarize_counts_scenes():
    infos = [SimpleNamespace(filename="th_0001/img/a.jpg", file_size=10),
             SimpleNamespace(filename="th_0001/img/b.jpg", file_size=10),
             SimpleNamespace(filename="th_0001/IPC/IPC_ENG/l.xlsx", file_size=5),
             SimpleNamespace(filename="th_0001/scan/x.las", file_size=999)]
    s = rz.summarize(infos)
    assert s["scenes"]["th_0001"] == {"photo": 2, "exhibit_list": 1}
    assert s["by_kind"]["blocked"] == (1, 999)
    assert "th_0001: 2 photos" in rz.format_summary(s)
