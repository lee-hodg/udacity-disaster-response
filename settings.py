import os


SITE_ROOT = os.path.dirname(os.path.realpath(__file__))

MODEL_PICKLE_FILENAME = os.path.join(SITE_ROOT, 'saved_model.pkl')

DATABASE_FILENAME = os.path.join(SITE_ROOT, 'DisasterResponse.db')

TABLE_NAME = 'disaster_messages'

MESSAGES_FILENAME = os.path.join(SITE_ROOT, 'data', 'disaster_messages.csv')

CATEGORIES_FILENAME = os.path.join(SITE_ROOT, 'data', 'disaster_categories.csv')
