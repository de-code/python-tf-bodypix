from tf_bodypix.utils.dist import get_required_and_extras


TENSORFLOW_REQUIREMENT = 'tensorflow==2.1.3'


class TestGetRequiredAndExtras:
    def test_should_group_single_requirement(self):
        assert get_required_and_extras(
            [('req1==1.2.3', ['group1'])]
        ) == (
            [],
            {'group1': ['req1==1.2.3'], 'all': ['req1==1.2.3']}
        )

    def test_should_fallback_to_default(self):
        assert get_required_and_extras(
            [('req1==1.2.3', [None])]
        ) == (
            ['req1==1.2.3'],
            {'all': ['req1==1.2.3']}
        )

    def test_should_group_multiple_requirement(self):
        assert get_required_and_extras(
            [('req1==1.2.3', ['group1']), ('req2==1.2.3', ['group2']), ('req3==1.2.3', [None])]
        ) == (
            ['req3==1.2.3'],
            {
                'group1': ['req1==1.2.3'],
                'group2': ['req2==1.2.3'],
                'all': ['req1==1.2.3', 'req2==1.2.3', 'req3==1.2.3']
            }
        )
