#include "multiviewergraphicsview.h"
#include <QWheelEvent>
void MultiviewerGraphicsView::wheelEvent(QWheelEvent *event)
{
    if(event->angleDelta().y() > 0)
    {
        scale(1.2, 1.2);
    }
    else
    {
        scale(1 / 1.2, 1 / 1.2);
    }
}
