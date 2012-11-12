#import <Cocoa/Cocoa.h>
#import <wchar.h>
#import <iostream>
#import <OpenGL/OpenGL.h>
#import <OpenGL/glu.h>
//
// Hooks back into the example code
//
extern void initOsd();
extern void updateGeom();
extern void display();

//
// Shared global state from the example
//
extern int g_width, g_height, g_frame;


//
// OSX application bootstrap
//
@class View;

@interface View : NSOpenGLView <NSWindowDelegate> {
    NSRect m_frameRect;
    BOOL m_didInit;
    uint64_t m_previousTime;
    NSTimer* m_timer;
}

- (void) animate;

@end

@implementation View

-(void)windowWillClose:(NSNotification *)note 
{
    [[NSApplication sharedApplication] terminate:self];
}

- (void) timerFired:(NSTimer*) timer
{
    [self animate];     
}

- (id) initWithFrame: (NSRect) frame
{
    m_didInit = FALSE;
    
    //
    // Various GL state, of note is the 3.2 Core profile selection
    // and 8x antialiasing, which we might want to disable for performance
    //
    int attribs[] = {
        NSOpenGLPFAAccelerated,
        NSOpenGLPFADoubleBuffer,
        NSOpenGLPFADepthSize, 24,
        NSOpenGLPFAAlphaSize, 8,
        NSOpenGLPFAColorSize, 32,
        NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersion3_2Core,
        NSOpenGLPFANoRecovery,
        kCGLPFASampleBuffers, 1,
        kCGLPFASamples, 8,
        0
    };

    NSOpenGLPixelFormat *fmt = [[NSOpenGLPixelFormat alloc]
                            initWithAttributes:(NSOpenGLPixelFormatAttribute*) attribs];

    self = [self initWithFrame:frame pixelFormat:fmt];
    [fmt release];
    m_frameRect = frame;
    m_timer = [[NSTimer
                       scheduledTimerWithTimeInterval:1.0f/120.0f
                       target:self 
                       selector:@selector(timerFired:)
                       userInfo:nil
                       repeats:YES] retain];

    return self;
}

- (void) drawRect:(NSRect) theRect
{
    if (!m_didInit) {
        initOsd();

        m_didInit = YES;
        
        [[self window] setLevel: NSFloatingWindowLevel];
        [[self window] makeKeyAndOrderFront: self];
        [[self window] setTitle: [NSString stringWithUTF8String: "SimpleCPU"]];
    }

    // run the example code
    display();
    
    [[self openGLContext] flushBuffer]; 
}

- (void) animate
{
    g_frame++;
    updateGeom();
    [self display];
}
@end

int main(int argc, const char *argv[])
{
    // 
    // It seems that some ATI cards fall back to software when rendering geometry shaders
    // but only *sometimes*. Post this notice so it's obvious that OSD isn't the problem.
    //
    // XXX(jcowles): we should only use vertex and fragment shaders to avoid this potential confusion
    //
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "NOTICE: Some Apple hardware runs geometry shaders in software, which will cause" << std::endl;
    std::cout << "        this demo to run extremely slowly. That slowness is not related to OSD" << std::endl;
    std::cout << "        and can be avoided by not using OpenGL geometry shaders." << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    NSAutoreleasePool *pool = [NSAutoreleasePool new];
    NSApplication *NSApp = [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    
    id menubar = [[NSMenu new] autorelease];
    id appMenuItem = [[NSMenuItem new] autorelease];
    [menubar addItem:appMenuItem];
    [NSApp setMainMenu:menubar];
    
    id appMenu = [[NSMenu new] autorelease];
    id appName = [[NSProcessInfo processInfo] processName];
    id quitTitle = [@"Quit " stringByAppendingString:appName];
    id quitMenuItem = [[[NSMenuItem alloc] initWithTitle:quitTitle
    action:@selector(terminate:) keyEquivalent:@"q"] autorelease];
    [appMenu addItem:quitMenuItem];
    [appMenuItem setSubmenu:appMenu];

    NSRect screenBounds = [[NSScreen mainScreen] frame];

    g_width = 512;
    g_height = 512;
    NSRect viewBounds = NSMakeRect(0, 0, g_width, g_height);
    
    View* view = [[View alloc] initWithFrame:viewBounds];
    
    NSRect centered = NSMakeRect(NSMidX(screenBounds) - NSMidX(viewBounds),
                                 NSMidY(screenBounds) - NSMidY(viewBounds),
                                 viewBounds.size.width, viewBounds.size.height);
    
    NSWindow *window = [[NSWindow alloc]
        initWithContentRect:centered
        styleMask:NSTitledWindowMask | NSClosableWindowMask | NSMiniaturizableWindowMask
        backing:NSBackingStoreBuffered
        defer:NO];

    [window setContentView:view];
    [window setDelegate:view];
    [window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];

    [view release];
    
    [NSApp run];


    [pool release];
    return EXIT_SUCCESS;
}


