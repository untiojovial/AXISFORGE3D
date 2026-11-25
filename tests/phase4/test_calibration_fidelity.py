import unittest
from research_grasping.phase4.fidelity_analysis import confusion, correlation, group_summary, classify_numerics


class FidelityTests(unittest.TestCase):
    def test_no_false_correlation_or_zero_division(self):
        self.assertIsNone(correlation([0,0,0],[0,1,0])['pearson'])
        self.assertIsNone(confusion([False,None],[True,True])['precision'])
        self.assertEqual(confusion([False,None],[True,True])['fn'],1)

    def test_valid_nominal_and_invalid_variant_are_separate(self):
        x={'f3_status':'completed_with_numerical_failures','f3':{'nominal':{'metrics':{'numerical_valid':True,'eligible_sustained':True,'sustained':True}},'numerics':[{'metrics':{'numerical_valid':False,'eligible_sustained':None}},{'metrics':{'numerical_valid':True,'eligible_sustained':True}}]}}
        out=classify_numerics(x)
        self.assertIs(out['eligible'],True)
        self.assertIs(out['numeric_changed'],False)
        self.assertEqual(out['numeric_invalid_variants'],1)
        x['f3']['nominal']['metrics']['numerical_valid']=False
        out=classify_numerics(x)
        self.assertIsNone(out['eligible'])
        self.assertIsNone(out['numeric_changed'])

    def test_missing_is_unknown_not_failure(self):
        rows=[dict(f1_status='rejected',f3_status='blocked',f0=False,f1=False,f2=0.,f2_positive=False,
                   raw=None,eligible=None,settle_stable=None,penetration=None,numeric_changed=None,nominal_valid=False,numeric_valid_variants=0,numeric_invalid_variants=0,native_elapsed_s=1.,f3_elapsed_s=None),
              dict(f1_status='feasible',f3_status='completed',f0=True,f1=True,f2=.1,f2_positive=True,
                   raw=True,eligible=True,settle_stable=True,penetration=.002,numeric_changed=True,nominal_valid=True,numeric_valid_variants=2,numeric_invalid_variants=1,native_elapsed_s=2.,f3_elapsed_s=3.)]
        report=group_summary(rows)
        self.assertEqual(report['unknown'],1)
        self.assertEqual(report['f1_vs_eligible']['n'],1)
        self.assertEqual(report['strict_penetration_eligible'],0)
        self.assertEqual(report['strategies']['always']['f3_unknown'],1)
        self.assertEqual(report['strategies']['f1']['reachable_success_retained'],1)
        self.assertEqual(report['strategies']['f1']['measured_nominal_f3_elapsed_s'],3.)


if __name__=='__main__':unittest.main()
