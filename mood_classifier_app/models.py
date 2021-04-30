from django.db import models

# Create your models here.
# Want this to be abstract
class UserEntry(models.Model):
    user_input_text = models.TextField(blank=True)
    entry_date = models.DateTimeField('date entered', blank=True)
    # one or zero
    mood_classification = models.IntegerField(blank=True)
