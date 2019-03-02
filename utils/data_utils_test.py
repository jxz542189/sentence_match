import unittest
from utils.data_util import read_csv


class DataUtilsTest(unittest.TestCase):  # 继承unittest.TestCase
    def tearDown(self):
        # 每个测试用例执行之后做操作
        print('case ending--------------\n')

    def setUp(self):
        # 每个测试用例执行之前做操作
        print('case starting--------------\n')

    @classmethod
    def tearDownClass(self):
        # 必须使用 @ classmethod装饰器, 所有test运行完后运行一次
        print('all cases ended--------------\n')

    @classmethod
    def setUpClass(self):
        # 必须使用@classmethod 装饰器,所有test运行前运行一次
        print('all cases started--------------\n')

    # def test_a_run(self):
    #     self.assertEqual(1, 1)  # 测试用例
    #
    # def test_b_run(self):
    #     self.assertEqual(2, 2)  # 测试用例
    #
    def test_read_csv(self):
        print("test read_csv function")
        file_name = "atec_nlp_sim_train_all.csv"
        data_csv = read_csv(file_name, N=5)
        print(data_csv)


if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例