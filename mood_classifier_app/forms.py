from django import forms
from mood_classifier_app.models import UserEntry


class UserEntryForm(forms.ModelForm):
    user_input_text = forms.CharField(widget=forms.Textarea, label='')
    class Meta:
        model = UserEntry
        exclude = ['entry_date', 'mood_classification']
