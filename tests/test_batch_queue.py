import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile
from ui_qt.main_window import BatchQueueItem, BatchQueueModel, MainWindow

class TestBatchQueue(unittest.TestCase):
    def test_queue_item_creation(self):
        item = BatchQueueItem(path="/path/to/file.mp3", display_name="file.mp3")
        self.assertEqual(item.path, "/path/to/file.mp3")
        self.assertEqual(item.display_name, "file.mp3")
        self.assertEqual(item.status, "queued")
        self.assertEqual(item.progress, 0.0)
        self.assertIsNone(item.error_message)

    def test_queue_model_add_and_exact_path_duplicates(self):
        model = BatchQueueModel()
        self.assertEqual(model.rowCount(), 0)
        
        # Add first item
        success = model.add_item("/path/to/a.mp3")
        self.assertTrue(success)
        self.assertEqual(model.rowCount(), 1)
        self.assertEqual(model.get_item(0).display_name, "a.mp3")
        
        # Add duplicate
        success = model.add_item("/path/to/a.mp3")
        self.assertFalse(success)
        self.assertEqual(model.rowCount(), 1)
        
        # Add another unique item
        success = model.add_item("/path/to/b.mp3")
        self.assertTrue(success)
        self.assertEqual(model.rowCount(), 2)

    def test_queue_model_allows_same_basename_from_different_folders(self):
        model = BatchQueueModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            first_dir = os.path.join(tmpdir, "first")
            second_dir = os.path.join(tmpdir, "second")
            os.makedirs(first_dir)
            os.makedirs(second_dir)
            first = os.path.join(first_dir, "capture.wav")
            second = os.path.join(second_dir, "capture.wav")
            with open(first, "w") as f:
                f.write("")
            with open(second, "w") as f:
                f.write("")

            self.assertTrue(model.add_item(first))
            self.assertTrue(model.add_item(second))
            self.assertFalse(model.add_item(first))

            self.assertEqual(model.rowCount(), 2)
            self.assertEqual(model.get_item(0).display_name, "first/capture.wav")
            self.assertEqual(model.get_item(1).display_name, "second/capture.wav")
            self.assertEqual(model.get_item(0).path, os.path.abspath(first))
            self.assertEqual(model.get_item(1).path, os.path.abspath(second))

    def test_queue_model_remove_and_clear(self):
        model = BatchQueueModel()
        model.add_item("1.mp3")
        model.add_item("2.mp3")
        model.add_item("3.mp3")
        self.assertEqual(model.rowCount(), 3)
        
        # Remove middle item
        model.remove_item(1) # removes 2.mp3
        self.assertEqual(model.rowCount(), 2)
        self.assertEqual(model.get_item(0).display_name, "1.mp3")
        self.assertEqual(model.get_item(1).display_name, "3.mp3")
        
        # Clear
        model.clear()
        self.assertEqual(model.rowCount(), 0)

    def test_queue_model_data_role(self):
        model = BatchQueueModel()
        model.add_item("test.mp3")
        model.update_item_status(0, "processing", progress=50.0)
        
        self.assertEqual(model.rowCount(), 1)
        item = model.get_item(0)
        self.assertEqual(item.status, "processing")
        self.assertEqual(item.progress, 50.0)

    def test_get_next_queued_index(self):
        model = BatchQueueModel()
        model.add_item("1.mp3")
        model.add_item("2.mp3")
        model.add_item("3.mp3")
        
        self.assertEqual(model.get_next_queued_index(), 0)
        
        model.update_item_status(0, "completed")
        self.assertEqual(model.get_next_queued_index(), 1)
        
        model.update_item_status(1, "failed")
        self.assertEqual(model.get_next_queued_index(), 2)
        
        model.update_item_status(2, "canceled")
        self.assertEqual(model.get_next_queued_index(), -1)

    def test_clear_completed(self):
        model = BatchQueueModel()
        model.add_item("1.mp3")
        model.add_item("2.mp3")
        model.add_item("3.mp3")
        
        model.update_item_status(0, "completed")
        model.update_item_status(1, "queued")
        model.update_item_status(2, "failed")
        
        model.clear_completed()
        self.assertEqual(model.rowCount(), 1)
        self.assertEqual(model.get_item(0).display_name, "2.mp3")

class TestBatchQueueMainWindowLogic(unittest.TestCase):
    def setUp(self):
        # We don't want to build a full MainWindow here as it requires a QApplication
        self.win = MagicMock(spec=MainWindow)
        self.win.batch_queue_model = BatchQueueModel()
        self.win.last_open_dir = "/tmp"
        
        # We'll allow the real methods to be tested on the mock object
        self.win._handle_incoming_paths = MainWindow._handle_incoming_paths.__get__(self.win, MainWindow)
        self.win._scan_folder_for_media = MainWindow._scan_folder_for_media.__get__(self.win, MainWindow)

    def test_scan_folder_for_media(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            a_mp3 = os.path.join(tmpdir, "a.mp3")
            b_txt = os.path.join(tmpdir, "b.txt")
            c_mp4 = os.path.join(tmpdir, "c.mp4")
            subfolder = os.path.join(tmpdir, "subfolder")
            
            os.makedirs(subfolder)
            with open(a_mp3, "w") as f: f.write("")
            with open(b_txt, "w") as f: f.write("")
            with open(c_mp4, "w") as f: f.write("")
            
            results = self.win._scan_folder_for_media(tmpdir)
            self.assertEqual(len(results), 2)
            self.assertIn(a_mp3, results)
            self.assertIn(c_mp4, results)

    def test_handle_incoming_paths_mixed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            direct_mp3 = os.path.join(tmpdir, "direct.mp3")
            folder = os.path.join(tmpdir, "folder")
            inner_wav = os.path.join(folder, "inner.wav")
            
            os.makedirs(folder)
            with open(direct_mp3, "w") as f: f.write("")
            with open(inner_wav, "w") as f: f.write("")
            
            paths = [direct_mp3, folder]
            
            self.win._handle_incoming_paths(paths)
            
            self.assertEqual(self.win.batch_queue_model.rowCount(), 2)
            self.assertEqual(self.win.batch_queue_model.get_item(0).display_name, "direct.mp3")
            self.assertEqual(self.win.batch_queue_model.get_item(1).display_name, "inner.wav")

if __name__ == "__main__":
    unittest.main()
