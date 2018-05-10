from openrec.recommenders import Recommender


class CombinedRecommender(Recommender):

    def __init__(self, batch_size, dim_embed, max_user, max_item, test_batch_size=None, l2_reg=None, opt='SGD', sess_config=None):
        pass

    def _build_user_inputs(self, train=True):

        if train:
            self._add_input(name='user_id', dtype='int32',
                            shape=[self._batch_size])
        else:
            self._add_input(name='user_id', dtype='int32',
                            shape=[None], train=False)

    def _build_item_inputs(self, train=True):

        if train:
            self._add_input(name='item_id', dtype='int32',
                            shape=[self._batch_size])
        else:
            self._add_input(name='item_id', dtype='none', train=False)

    # this is for contextual data
    def _build_extra_inputs(self, train=True):
        pass
        # if train:
        #     self._add_input(name='lables', dtype='float32', shape=[self._batch_size])

    '''
    Define input mappings
    '''

    def _input_mappings(self, batch_data, train):

        if train:
            return {self._get_input('user_id'): batch_data['user_id_input'],
                    self._get_input('item_id'): batch_data['item_id_input'],
                    self._get_input('labels'): batch_data['labels']}
        else:
            return {self._get_input('user_id', train=False): batch_data['user_id_input']}

    '''
    Define modules using `add_module` function.
    '''

    def _build_user_extractions(self, train=True):
        pass

    def _build_item_extractions(self, train=True):

        self._add_module('CUSTOM_MODULE',
                         None
                         # our custom model goes here: http://openrec.readthedocs.io/en/latest/modules/openrec.modules.module.html
                         )

    def _build_default_interactions(self, train=True):
        pass

    def _build_serving_graph(self):
        self._scores = self._get_module(
            'CUSTOM_MODULE', train=False).get_outputs()[0]
