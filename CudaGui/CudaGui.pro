QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    geotiffwrapper.cpp \
    imagelabel.cpp \
    loaddialog.cpp \
    main.cpp \
    mainwindow.cpp \
    multiviewer.cpp \
    multiviewergraphicsview.cpp \
    tiffview.cpp

HEADERS += \
    declares.h \
    geotiffwrapper.h \
    imagelabel.h \
    loaddialog.h \
    mainwindow.h \
    multiviewer.h \
    multiviewergraphicsview.h \
    tiffview.h

FORMS += \
    loaddialog.ui \
    mainwindow.ui \
    multiviewer.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

win32: LIBS += -L$$PWD/../../../CUDA/dev/lib/x64/ -lcudart

INCLUDEPATH += $$PWD/../../../CUDA/dev/lib/x64
DEPENDPATH += $$PWD/../../../CUDA/dev/lib/x64

win32: LIBS += -L$$PWD/../tiff-4.1.0/libtiff/ -llibtiff

INCLUDEPATH += $$PWD/../tiff-4.1.0/libtiff
DEPENDPATH += $$PWD/../tiff-4.1.0/libtiff

win32: LIBS += -L$$PWD/../libgeotiff-master/libgeotiff/ -lgeotiff

INCLUDEPATH += $$PWD/../libgeotiff-master/libgeotiff
INCLUDEPATH += $$PWD/../libgeotiff-master/libgeotiff/libxtiff
DEPENDPATH += $$PWD/../libgeotiff-master/libgeotiff

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../CudaBackEnd/x64/release/ -lCudaBackEnd
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../CudaBackEnd/x64/debug/ -lCudaBackEnd

INCLUDEPATH += $$PWD/../CudaBackEnd/x64/Debug
DEPENDPATH += $$PWD/../CudaBackEnd/x64/Debug

RESOURCES += \
    Resources.qrc
