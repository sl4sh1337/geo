#include "imagelabel.h"

void ImageLabel::paintEvent(QPaintEvent *event)
{
    QLabel::paintEvent(event);
    if(resized)
    {
        move(*label_pos);
        resized = false;
    }
}
