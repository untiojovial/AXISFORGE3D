import unittest
import numpy as np
from research_grasping.phase3.prepare_pilot import catalogue, mesh_points, selected_objects


def xyz(x,y,z):
    return {'x':x,'y':y,'z':z}


def rec(i):
    return {'id':i,'candidate':{'center':xyz(.45,0,.42),'approach':xyz(0,0,1),
            'closing':xyz(1,0,0),'width':.05,'contacts':[{'point':xyz(.46,0,.42),'normal':xyz(1,0,0)}]},
            'plan':None,'rejection':None,'quality':None,'qualityElapsedS':None,'attemptElapsedS':None}


class PreparePilotTests(unittest.TestCase):
    def test_reject_test_and_duplicates(self):
        m={'objects':[{'object_id':'a','split':'train'},{'object_id':'b','split':'test'}]}
        self.assertEqual(len(selected_objects(m,['a'])),1)
        with self.assertRaises(ValueError): selected_objects(m,['a','b'])
        with self.assertRaises(ValueError): selected_objects(m,['a','a'])

    def test_pre_evaluation_and_uniform_ids(self):
        rows=[rec(i) for i in range(100)]
        ids, f, cs = catalogue(rows, 4)
        self.assertEqual(ids,['0','33','66','99'])
        self.assertEqual(len(f[0]),16)
        self.assertAlmostEqual(f[0][9],.5)
        self.assertEqual(cs['0']['contacts'][0],[.46,0,.42])
        rows[0]['quality']={'score':1}
        with self.assertRaises(ValueError): catalogue(rows, 4)

    def test_mesh_alignment_and_reproducibility(self):
        triangles=np.array([[[2.,3.,4.],[4.,3.,4.],[2.,7.,4.]],[[2.,3.,4.],[4.,3.,4.],[2.,3.,6.]]])
        a, low, high, origin=mesh_points(triangles)
        b, _, _, _=mesh_points(triangles)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(origin,[3,5,4])
        self.assertTrue(np.all(a.min(axis=0) >= np.array([-.55,-2.,.42])-1e-12))
        self.assertEqual(a.shape,(256,3))


if __name__=='__main__': unittest.main()
