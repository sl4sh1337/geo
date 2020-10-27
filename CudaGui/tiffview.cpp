#include "tiffview.h"
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsScene>
#include <QGuiApplication>

void TIFFView::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if(event->buttons() == Qt::MouseButton::LeftButton)
    {
        if(!(QGuiApplication::keyboardModifiers().testFlag(Qt::ControlModifier)))
            scene()->clearSelection();

        setSelected(true);
    }
    setOpacity(0.5);
}

void TIFFView::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    QGraphicsPixmapItem::mouseReleaseEvent(event);
    setSelected(true);
    setOpacity(1.0);
}
