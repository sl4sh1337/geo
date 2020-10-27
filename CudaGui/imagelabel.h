#ifndef IMAGELABEL_H
#define IMAGELABEL_H

#include <QLabel>

class ImageLabel : public QLabel
{
public:
    bool resized = false;
    QPoint* label_pos = nullptr;
    // QWidget interface
protected:
    void paintEvent(QPaintEvent *event);
};

#endif // IMAGELABEL_H
