import telepot
from time import time
from functools import wraps

bot = telepot.Bot('')
response = bot.getUpdates()
botid = ''


def send_message(z):
    try:
        bot.sendMessage(botid, z)
    except Exception as e:
        print('Error sending message: %s' % e)


def send_photo(f):
    try:
        bot.sendPhoto(botid, f)
    except Exception as e:
        print('Error sending photo: %s' % e)


class computation_time:
    def __init__(self, *args):
        self.func_args = args

    def __call__(self, func):
        @wraps(func)
        def func_wrapper(*args, **kwargs):
            start = time()
            res = func(*args, **kwargs)
            stop = time()
            t = stop - start
            mins, secs = divmod(t, 60)
            hours, mins = divmod(mins, 60)
            params = ''
            for arg, argval in kwargs.items():
                if arg in self.func_args:
                    params += '%s=%s, ' % (arg, str(argval))
            fname = '%s(%s)' % (func.__name__, params[:-2])
            send_message('Computation time for %s: %d h. %d m. %d s.' % (fname, hours, mins, secs))
            return res
        return func_wrapper
